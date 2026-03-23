"""
591 租屋網爬蟲 — 台南市 × 套房
用途：收集 input/output pair，供 Gemini fine-tune / Qwen2.5 LoRA 訓練資料使用
頻率：3–5 秒/request（避免對伺服器造成負擔）
法律備註：僅供個人研究與 AI 訓練使用，請勿商業用途，嚴格控制頻率。

使用方式：
    python scripts/ad_copywriter/01_scrape_591.py
    python scripts/ad_copywriter/01_scrape_591.py --region 15 --kind 2 --max 300

region code（常見）：
    15 = 台南市（預設）
    1  = 台北市
    3  = 台中市
    6  = 高雄市

kind code：
    0 = 不限
    1 = 整層住家
    2 = 獨立套房（預設）
    3 = 分租套房
    4 = 雅房
"""

import argparse
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("591_scraper")

# ── 預設設定 ──────────────────────────────────────────────────────
DEFAULT_REGION = 15   # 台南市
DEFAULT_KIND   = 2    # 獨立套房
DEFAULT_MAX    = 300
DELAY_MIN      = 3.0
DELAY_MAX      = 5.5

ROOT_DIR  = Path(__file__).resolve().parents[2]  # Room_Inspector/
RAW_DIR   = ROOT_DIR / "data" / "raw" / "591"
NORM_DIR  = ROOT_DIR / "data" / "normalized"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Referer": "https://rent.591.com.tw/",
}

# ── 欄位對照表 ─────────────────────────────────────────────────────
APPLIANCE_MAP = {
    "cold":       "冷氣",
    "washer":     "洗衣機",
    "icebox":     "冰箱",
    "hotwater":   "熱水器",
    "tv":         "電視",
    "broadband":  "網路",
    "naturalgas": "天然瓦斯",
}
FURNITURE_MAP = {
    "bed":     "床架",
    "wardrobe":"衣櫃",
    "sofa":    "沙發",
    "table":   "桌椅",
    "desk":    "書桌",
}
BATH_TYPE_MAP = {0: "共用衛浴", 1: "獨立衛浴", 2: "獨立衛浴"}
KITCHEN_MAP   = {0: "無廚房",   1: "有廚房",   2: "有廚房"}
DEPOSIT_MAP   = {0: "", 1: "1個月", 2: "2個月", 3: "3個月"}
MIN_LEASE_MAP = {0: "", 1: "1個月", 3: "3個月", 6: "6個月", 12: "1年"}


# ── Session 初始化 ─────────────────────────────────────────────────
def make_session() -> requests.Session:
    """先打首頁，取得 csrf-token 與 cookie（包含 deviceid）。"""
    sess = requests.Session()
    sess.headers.update(HEADERS)

    resp = sess.get("https://rent.591.com.tw/", timeout=15)
    resp.raise_for_status()

    m = re.search(r'<meta name="csrf-token" content="([^"]+)"', resp.text)
    if not m:
        raise RuntimeError(
            "找不到 csrf-token，591 頁面結構可能已更新 → 改用 Playwright 版本"
        )
    csrf = m.group(1)
    sess.headers.update({"X-CSRF-TOKEN": csrf})

    # 台南市需設定 urlJumpIp cookie，否則區域篩選無效
    # 台南市 = 20，台北市 = 1（依 591 城市 IP jump 值）
    sess.cookies.set("urlJumpIp", "20", domain=".591.com.tw", path="/")
    log.info(f"Session ready, csrf={csrf[:20]}...")
    return sess


# ── 列表 API ───────────────────────────────────────────────────────
def fetch_list(sess: requests.Session, region: int, kind: int, first_row: int = 0) -> dict:
    url = "https://rent.591.com.tw/home/search/rsList"
    params = {
        "is_format_data": 1,
        "is_new_list":    1,
        "type":           1,
        "region":         region,
        "kind":           kind,
        "firstRow":       first_row,
        "order":          "posttime",
        "orderType":      "desc",
    }
    resp = sess.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── 詳情 API ───────────────────────────────────────────────────────
def fetch_detail(sess: requests.Session, house_id: str) -> Optional[dict]:
    deviceid = (
        sess.cookies.get("T591_TOKEN", "")
        or sess.cookies.get("deviceid", "pc_default")
    )
    url = "https://bff.591.com.tw/v1/house/rent/detail"
    params = {"id": house_id, "device": "pc", "deviceid": deviceid}
    try:
        resp = sess.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"detail fetch failed for id={house_id}: {e}")
        return None


# ── Normalize ─────────────────────────────────────────────────────
def normalize(list_item: dict, detail: Optional[dict]) -> dict:
    """把 591 原始資料 map 成統一 schema。"""
    house     = (detail or {}).get("data", {}).get("house", {})
    tags_raw  = (detail or {}).get("data", {}).get("tags", [])

    tag_values = {t.get("value", "") for t in tags_raw}
    appliances = [v for k, v in APPLIANCE_MAP.items() if k in tag_values]
    furniture  = [v for k, v in FURNITURE_MAP.items() if k in tag_values]

    # 非設備/家具的標籤視為房源特色
    known_keys = set(APPLIANCE_MAP) | set(FURNITURE_MAP)
    feature_tags = [
        t.get("name", "")
        for t in tags_raw
        if t.get("value", "") not in known_keys and t.get("name", "")
    ]

    # 費用
    water_raw  = house.get("water_bind", -1)
    elec_raw   = house.get("electricity_bind", -1)
    water_fee  = "費用內含" if water_raw == 1 else "台水計費" if water_raw == 0 else ""
    elec_fee   = "費用內含" if elec_raw  == 1 else "台電計費" if elec_raw  == 0 else ""

    # 樓層
    fl  = list_item.get("floor", "") or house.get("floor", "")
    tfl = list_item.get("total_floor", "") or house.get("total_floor", "")
    floor_str = f"{fl}/{tfl}" if fl and tfl else str(fl or "")

    # 廣告文案（訓練 output）
    raw_desc  = house.get("desc", "").strip()
    raw_title = list_item.get("title", "").strip()

    # 附近站點
    transport_info = ""
    transportation = (detail or {}).get("data", {}).get("transportation", [])
    if transportation:
        t = transportation[0]
        transport_info = f"近{t.get('name', '')} {t.get('distance', '')}公尺"

    return {
        # meta
        "source_site": "591",
        "source_url":  f"https://rent.591.com.tw/home/detail/{list_item.get('post_id', '')}",
        "source_id":   str(list_item.get("post_id", "")),
        "scraped_at":  datetime.utcnow().isoformat(),
        # 核心欄位（與 generate_listing_descriptions 的 ListingDescRequest 對齊）
        "city":           list_item.get("region_name", "").replace("台", "臺"),
        "district":       list_item.get("section_name", ""),
        "layout":         list_item.get("kind_name", ""),
        "size":           float(list_item.get("area", 0) or 0),
        "floor":          floor_str,
        "price":          int(list_item.get("price", 0) or 0),
        "deposit":        DEPOSIT_MAP.get(int(house.get("deposit_type", 0) or 0), ""),
        "appliances":     appliances,
        "furniture":      furniture,
        "bathType":       BATH_TYPE_MAP.get(int(house.get("bath_type", -1) or -1), ""),
        "kitchenType":    KITCHEN_MAP.get(int(house.get("kitchen", -1) or -1), ""),
        "petAllowed":     "pet" in tag_values,
        "cookingAllowed": "cook" in tag_values,
        "waterFee":       water_fee,
        "electricFee":    elec_fee,
        "minRental":      MIN_LEASE_MAP.get(int(house.get("min_lease", 0) or 0), ""),
        "tags":           feature_tags,
        "nearby":         transport_info or list_item.get("address", ""),
        "condition":      house.get("condition_summary", ""),
        # 訓練 pair 用
        "raw_title":  raw_title,
        "raw_desc":   raw_desc,
        # 品質評分（0–100）
        "alignment_score": _score(raw_desc, appliances, list_item),
    }


def _score(raw_desc: str, appliances: list, item: dict) -> int:
    """簡易對齊分數：欄位越齊 + 文案越完整 → 分越高。"""
    score = 0
    if item.get("price"):          score += 20
    if item.get("area"):           score += 10
    if item.get("section_name"):   score += 10
    if appliances:                 score += 10
    if raw_desc and len(raw_desc) >= 30:  score += 30
    if raw_desc and len(raw_desc) >= 80:  score += 20
    return min(score, 100)


# ── 主流程 ─────────────────────────────────────────────────────────
def run(region: int, kind: int, max_rows: int):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    sess = make_session()
    normalized_all = []
    first_row = 0
    total_fetched = 0

    while total_fetched < max_rows:
        log.info(f"Fetching list firstRow={first_row} ...")
        data = fetch_list(sess, region, kind, first_row)

        # 格式異常（加密或被擋）
        inner = data.get("data", {})
        if not isinstance(inner, dict) or "data" not in inner:
            log.warning("列表回傳格式異常，可能已加密或被擋。停止。")
            log.warning(f"Response keys: {list(data.keys())}")
            log.warning("建議：改用 Playwright 版本（02_scrape_591_playwright.py）")
            break

        items = inner["data"]
        if not items:
            log.info("無更多資料，結束。")
            break

        for item in items:
            house_id = str(item.get("post_id", ""))
            if not house_id:
                continue

            # 存原始列表
            (RAW_DIR / f"list_{house_id}.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            detail = fetch_detail(sess, house_id)

            if detail:
                (RAW_DIR / f"detail_{house_id}.json").write_text(
                    json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            norm = normalize(item, detail)
            normalized_all.append(norm)
            total_fetched += 1
            log.info(
                f"  [{total_fetched:>4}] id={house_id} "
                f"score={norm['alignment_score']:>3} "
                f"desc_len={len(norm['raw_desc']):>4}"
            )

            if total_fetched >= max_rows:
                break

        first_row += 30
        total_records = int(inner.get("records", 0))
        if total_records and first_row >= total_records:
            break

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # 儲存 normalized JSONL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = NORM_DIR / f"591_region{region}_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in normalized_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    high_quality = sum(1 for r in normalized_all if r["alignment_score"] >= 70)
    log.info(f"完成。總共 {len(normalized_all)} 筆 → {out_path}")
    log.info(f"高品質樣本（score≥70）: {high_quality} 筆")
    log.info(f"下一步：python scripts/ad_copywriter/02_build_pairs.py --input {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="591 租屋網爬蟲")
    parser.add_argument("--region", type=int, default=DEFAULT_REGION, help="城市代碼（預設 15=台南市）")
    parser.add_argument("--kind",   type=int, default=DEFAULT_KIND,   help="房型代碼（預設 2=套房）")
    parser.add_argument("--max",    type=int, default=DEFAULT_MAX,    help="最多抓幾筆（預設 300）")
    args = parser.parse_args()

    run(region=args.region, kind=args.kind, max_rows=args.max)
