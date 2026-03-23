"""
本機推論腳本：用訓練好的 LoRA adapter 產生三版廣告文案
用途：測試 adapter 品質、與 Gemini baseline 對比

使用方式：
    # 互動模式（從 stdin 輸入 JSON）
    python scripts/ad_copywriter/04_inference.py

    # 批次模式
    python scripts/ad_copywriter/04_inference.py --input data/pairs/pairs_20260313_1200.jsonl --n 10
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference")

ROOT_DIR    = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT_DIR / "checkpoints" / "ad_copywriter" / "lora-qwen2.5-1.5b" / "final_adapter"

SYSTEM_PROMPT = "你是台灣租屋廣告文案師，輸出克制、地道的繁體中文。"

EXAMPLE_LISTING = {
    "city": "臺南市",
    "district": "仁德區",
    "layout": "獨立套房",
    "size": 10.5,
    "floor": "3/5",
    "price": 8500,
    "deposit": "2個月",
    "appliances": ["冷氣", "洗衣機", "冰箱", "網路"],
    "furniture": ["床架", "衣櫃", "書桌"],
    "bathType": "獨立衛浴",
    "kitchenType": "有廚房",
    "petAllowed": False,
    "cookingAllowed": True,
    "waterFee": "台水計費",
    "electricFee": "台電計費",
    "minRental": "6個月",
    "tags": ["近捷運", "採光佳"],
    "nearby": "近仁德車站 300公尺",
    "condition": "",
}


def load_model(adapter_dir: Path):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(f"缺少套件：{e}\n請執行：pip install transformers peft bitsandbytes") from e

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"找不到 adapter：{adapter_dir}\n請先執行 03_finetune_qwen.py"
        )

    log.info(f"載入 adapter：{adapter_dir}")
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, listing: dict, max_new_tokens: int = 300) -> str:
    import torch

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": json.dumps(listing, ensure_ascii=False)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_mode(model, tokenizer):
    print("\n=== 互動模式：輸入房源 JSON，按兩次 Enter 生成文案 ===")
    print(f"範例輸入：\n{json.dumps(EXAMPLE_LISTING, ensure_ascii=False, indent=2)}\n")
    while True:
        lines = []
        print(">>> 輸入 listing JSON（或 q 退出）：")
        while True:
            line = input()
            if line.lower() == "q":
                return
            lines.append(line)
            if not line:
                break
        raw = "\n".join(lines).strip()
        if not raw:
            continue
        try:
            listing = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤：{e}")
            continue
        print("\n生成中...")
        result = generate(model, tokenizer, listing)
        print(f"\n{'='*60}\n{result}\n{'='*60}\n")


def batch_mode(model, tokenizer, input_path: str, n: int):
    path = Path(input_path)
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records = records[:n]
    log.info(f"批次推論 {len(records)} 筆...")

    for i, r in enumerate(records, 1):
        # 從 messages 中取出 user 的 listing JSON
        listing_str = next(
            (m["content"] for m in r.get("messages", []) if m["role"] == "user"), "{}"
        )
        listing = json.loads(listing_str)
        ref_desc = next(
            (m["content"] for m in r.get("messages", []) if m["role"] == "assistant"), ""
        )

        generated = generate(model, tokenizer, listing)
        print(f"\n[{i}/{len(records)}] source_id={r.get('_meta', {}).get('source_id', '')}")
        print(f"  【參考文案】{ref_desc[:80]}...")
        print(f"  【生成文案】{generated[:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA adapter 推論")
    parser.add_argument("--adapter", default="",   help="adapter 路徑（預設自動找）")
    parser.add_argument("--input",   default="",   help="批次模式：pairs JSONL 路徑")
    parser.add_argument("--n",       type=int, default=5, help="批次模式：測試幾筆（預設 5）")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter) if args.adapter else ADAPTER_DIR
    model, tokenizer = load_model(adapter_dir)

    if args.input:
        batch_mode(model, tokenizer, args.input, args.n)
    else:
        interactive_mode(model, tokenizer)
