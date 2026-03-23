"""
Qwen2.5-1.5B-Instruct LoRA 訓練腳本（本機 RTX 3060 12GB）
輸入：data/pairs/pairs_*.jsonl
輸出：checkpoints/ad_copywriter/lora-qwen2.5-1.5b/

硬體：RTX 3060 12GB，FP16，QLoRA 4bit
預估訓練時間：300 樣本 × 3 epoch ≈ 15–30 分鐘

使用方式：
    python scripts/ad_copywriter/03_finetune_qwen.py
    python scripts/ad_copywriter/03_finetune_qwen.py --data data/pairs/pairs_20260313_1200.jsonl
    python scripts/ad_copywriter/03_finetune_qwen.py --epochs 5 --batch 4

依賴安裝：
    pip install transformers peft trl bitsandbytes datasets accelerate
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune_qwen")

ROOT_DIR   = Path(__file__).resolve().parents[2]
PAIRS_DIR  = ROOT_DIR / "data" / "pairs"
OUTPUT_DIR = ROOT_DIR / "checkpoints" / "ad_copywriter" / "lora-qwen2.5-1.5b"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load_latest_pairs(data_path: str) -> Path:
    if data_path:
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"找不到資料檔：{p}")
        return p
    candidates = sorted(PAIRS_DIR.glob("pairs_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"找不到訓練資料（{PAIRS_DIR}/pairs_*.jsonl），"
            "請先執行 02_build_pairs.py"
        )
    return candidates[-1]  # 最新的


def train(data_path: str, epochs: int, batch_size: int, lr: float):
    # lazy import（避免沒裝套件時直接 crash）
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        log.error(f"缺少套件：{e}")
        log.error("請執行：pip install transformers peft trl bitsandbytes datasets accelerate")
        raise

    pairs_file = load_latest_pairs(data_path)
    log.info(f"載入資料：{pairs_file}")

    # ── 載入資料 ──────────────────────────────────────────────────
    raw_records = []
    with pairs_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                raw_records.append(r)

    log.info(f"共 {len(raw_records)} 筆訓練樣本")

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 格式化成 chat template ──────────────────────────────────────
    def format_messages(record: dict) -> str:
        messages = record["messages"]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    texts = [format_messages(r) for r in raw_records]
    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=min(0.1, 30 / len(texts)), seed=42)
    log.info(f"train={len(split['train'])} eval={len(split['test'])}")

    # ── QLoRA（4-bit quantization）─────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    log.info(f"載入模型：{MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA 設定 ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 訓練設定 ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    log.info("開始訓練...")
    trainer.train()

    # ── 儲存 LoRA adapter ──────────────────────────────────────────
    adapter_path = OUTPUT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info(f"LoRA adapter 已儲存 → {adapter_path}")
    log.info("下一步：python scripts/ad_copywriter/04_inference.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B LoRA 訓練")
    parser.add_argument("--data",   default="",    help="指定 pairs JSONL（預設用最新的）")
    parser.add_argument("--epochs", type=int,   default=3,    help="訓練 epochs（預設 3）")
    parser.add_argument("--batch",  type=int,   default=2,    help="batch size（預設 2）")
    parser.add_argument("--lr",     type=float, default=2e-4, help="learning rate（預設 2e-4）")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch, args.lr)
