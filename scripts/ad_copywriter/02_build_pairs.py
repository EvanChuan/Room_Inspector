"""
清洗 + 配對腳本：normalized JSONL → training pairs JSONL
輸入：data/normalized/591_*.jsonl
輸出：data/pairs/pairs_YYYYMMDD.jsonl（Gemini / Qwen2.5 SFT 格式）

篩選條件（預設）：
  - alignment_score >= 70
  - raw_desc 長度 30–300 字
  - 去除個資（手機/地址）
  - 去重（SimHash）

使用方式：
    python scripts/ad_copywriter/02_build_pairs.py
    python scripts/ad_copywriter/02_build_pairs.py --input data/normalized/591_region15_20260313_1200.jsonl
    python scripts/ad_copywriter/02_build_pairs.py --min-score 60 --min-desc 50
"""

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_pairs")

ROOT_DIR  = Path(__file__).resolve().parents[2]
NORM_DIR  = ROOT_DIR / "data" / "normalized"
PAIRS_DIR = ROOT_DIR / "data" / "pairs"

# 個資 regex
_PHONE_RE   = re.compile(r"09\d{2}[-\s]?\d{3}[-\s]?\d{3}")
_PHONE2_RE  = re.compile(r"0[2-8]\d[-\s]?\d{7,8}")
_EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_LINE_RE    = re.compile(r"Line\s*[:：]?\s*\S+", re.IGNORECASE)

# 禁用詞
BANNED_WORDS = ["超值", "物超所值", "CP值", "頂樓加蓋", "秒殺", "佛心"]


def remove_pii(text: str) -> str:
    text = _PHONE_RE.sub("[電話]", text)
    text = _PHONE2_RE.sub("[電話]", text)
    text = _EMAIL_RE.sub("[信箱]", text)
    text = _LINE_RE.sub("[LINE]", text)
    return text


def simhash(text: str) -> str:
    """簡易 SimHash（64-bit）：用 unigram + bigram 特徵。"""
    tokens = list(text) + [text[i:i+2] for i in range(len(text)-1)]
    bits = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i in range(64):
        if bits[i] > 0:
            fingerprint |= (1 << i)
    return format(fingerprint, "016x")


def hamming(a: str, b: str) -> int:
    x = int(a, 16) ^ int(b, 16)
    return bin(x).count("1")


def to_training_record(norm: dict) -> dict:
    """轉換成 Gemini / Qwen2.5 SFT JSONL 格式。"""
    listing_input = {k: v for k, v in norm.items()
                     if k not in ("raw_title", "raw_desc", "alignment_score",
                                  "source_site", "source_url", "scraped_at")}

    return {
        "messages": [
            {
                "role": "system",
                "content": "你是台灣租屋廣告文案師，輸出克制、地道的繁體中文。"
            },
            {
                "role": "user",
                "content": json.dumps(listing_input, ensure_ascii=False)
            },
            {
                "role": "assistant",
                "content": norm["raw_desc"]
            },
        ],
        # 保留 meta，方便 debug
        "_meta": {
            "source_id":       norm.get("source_id", ""),
            "source_url":      norm.get("source_url", ""),
            "alignment_score": norm.get("alignment_score", 0),
            "desc_len":        len(norm.get("raw_desc", "")),
        }
    }


def process_file(path: Path, min_score: int, min_desc: int, max_desc: int) -> list[dict]:
    records = []
    skipped = {"score": 0, "short": 0, "long": 0, "banned": 0, "dup": 0}
    seen_hashes: dict[str, str] = {}  # hash → source_id

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                norm = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. 品質分數
            if norm.get("alignment_score", 0) < min_score:
                skipped["score"] += 1
                continue

            raw_desc = norm.get("raw_desc", "")

            # 2. 長度
            if len(raw_desc) < min_desc:
                skipped["short"] += 1
                continue
            if len(raw_desc) > max_desc:
                skipped["long"] += 1
                continue

            # 3. 禁用詞
            if any(w in raw_desc for w in BANNED_WORDS):
                skipped["banned"] += 1
                continue

            # 4. 去個資
            clean_desc = remove_pii(raw_desc)
            norm["raw_desc"] = clean_desc

            # 5. 去重（SimHash Hamming distance < 4 視為重複）
            sh = simhash(clean_desc)
            is_dup = False
            for prev_hash in seen_hashes:
                if hamming(sh, prev_hash) < 4:
                    is_dup = True
                    break
            if is_dup:
                skipped["dup"] += 1
                continue
            seen_hashes[sh] = norm.get("source_id", "")

            records.append(to_training_record(norm))

    log.info(
        f"{path.name}: 保留 {len(records)} 筆，"
        f"跳過 score={skipped['score']} short={skipped['short']} "
        f"long={skipped['long']} banned={skipped['banned']} dup={skipped['dup']}"
    )
    return records


def run(input_glob: str, min_score: int, min_desc: int, max_desc: int):
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    if input_glob:
        paths = list(Path(input_glob).parent.glob(Path(input_glob).name)) \
                if "*" in input_glob else [Path(input_glob)]
    else:
        paths = sorted(NORM_DIR.glob("591_*.jsonl"))

    if not paths:
        log.error(f"找不到輸入檔案（{input_glob or NORM_DIR / '591_*.jsonl'}）")
        return

    all_records = []
    for p in paths:
        all_records.extend(process_file(p, min_score, min_desc, max_desc))

    if not all_records:
        log.warning("沒有符合條件的樣本，請降低 --min-score 或 --min-desc")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = PAIRS_DIR / f"pairs_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"完成。{len(all_records)} 筆訓練樣本 → {out_path}")
    log.info(f"下一步：python scripts/ad_copywriter/03_finetune_qwen.py --data {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗 normalized JSONL → training pairs")
    parser.add_argument("--input",     default="",  help="指定輸入 JSONL（預設掃 data/normalized/591_*.jsonl）")
    parser.add_argument("--min-score", type=int, default=70,  help="alignment_score 門檻（預設 70）")
    parser.add_argument("--min-desc",  type=int, default=30,  help="raw_desc 最短字數（預設 30）")
    parser.add_argument("--max-desc",  type=int, default=300, help="raw_desc 最長字數（預設 300）")
    args = parser.parse_args()

    run(args.input, args.min_score, args.min_desc, args.max_desc)
