#!/usr/bin/env bash
# =============================================================================
# 01_download_datasets.sh
# 下載室內屋況缺陷偵測訓練用資料集
#
# 涵蓋資料集：
#   1. BD3      — 建築缺陷 sample（僅 37 張，完整集未公開，作參考用）
#   2. SDNET2018 — 混凝土裂縫（wall/deck/pavement，56,000 張）
#   3. DAGM 2007 — 工業表面缺陷（需手動從 Zenodo 下載）
#
# 使用方式：
#   bash scripts/01_download_datasets.sh
#
# 執行前請確認：
#   - git 已安裝（下載 BD3 sample）
#   - wget 或 curl 已安裝（下載 SDNET2018）
#   - 約 1.5 GB 以上磁碟空間
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_ROOT/data/raw"

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${GREEN}════════════════════════════════════════${NC}"; echo -e "${GREEN} $*${NC}"; echo -e "${GREEN}════════════════════════════════════════${NC}"; }

# =============================================================================
# 1. BD3 Dataset
# =============================================================================
section "BD3 Dataset（建築缺陷）"

BD3_DIR="$RAW_DIR/bd3"

warn "⚠ BD3 完整資料集（3,965 張）尚未公開發布"
warn "  GitHub repo 只含 37 張 sample 圖片（每類 5~7 張）"
warn "  完整資料需聯繫作者：https://github.com/Praveenkottari/BD3-Dataset"
echo ""

if [ -d "$BD3_DIR/.git" ]; then
    info "BD3 sample 已存在，執行 git pull 更新..."
    git -C "$BD3_DIR" pull --quiet
else
    info "正在 git clone BD3 sample（僅含示例圖片）..."
    git clone --depth=1 https://github.com/Praveenkottari/BD3-Dataset.git "$BD3_DIR"
    info "BD3 sample 下載完成：$BD3_DIR"
fi

# 顯示 BD3 類別統計
info "BD3 sample 類別統計（參考用）："
CLASSES_PATH="$BD3_DIR/sample images/class_images"
if [ -d "$CLASSES_PATH" ]; then
    for cls_dir in "$CLASSES_PATH"/*/; do
        cls_name=$(basename "$cls_dir")
        count=$(find "$cls_dir" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        echo "  - $cls_name: $count 張"
    done
    echo ""
    warn "這些 sample 圖片可作為「各類缺陷長什麼樣」的視覺參考，"
    warn "但數量不足以訓練，主要訓練資料來自 SDNET2018 + DAGM + 自拍照"
else
    warn "找不到 BD3 類別資料夾，路徑可能已更改，請手動確認：$CLASSES_PATH"
fi

# =============================================================================
# 2. SDNET2018 Dataset
# =============================================================================
section "SDNET2018 Dataset（混凝土裂縫，56,000 張）"

SDNET_DIR="$RAW_DIR/sdnet2018"
SDNET_ZIP="$SDNET_DIR/SDNET2018.zip"
# 正確 URL：filename=2（filename=0 已 HTTP 410 Gone）
SDNET_URL="https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional"

# 判斷是否已解壓
if [ -d "$SDNET_DIR/D" ] && [ -d "$SDNET_DIR/W" ] && [ -d "$SDNET_DIR/P" ]; then
    info "SDNET2018 已解壓，略過下載。"
else
    # 若已存在但損壞的舊 ZIP，先刪除再重下
    if [ -f "$SDNET_ZIP" ]; then
        zip_size=$(stat -c%s "$SDNET_ZIP" 2>/dev/null || echo 0)
        if [ "$zip_size" -lt 10000000 ]; then   # 小於 10MB 視為損壞
            warn "偵測到損壞的舊 ZIP（${zip_size} bytes），刪除後重新下載..."
            rm -f "$SDNET_ZIP"
        else
            info "SDNET2018 壓縮檔已存在（$(( zip_size / 1024 / 1024 )) MB），略過下載。"
        fi
    fi

    if [ ! -f "$SDNET_ZIP" ]; then
        info "正在下載 SDNET2018（約 504 MB）..."
        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "$SDNET_ZIP" "$SDNET_URL" || {
                warn "wget 下載失敗，嘗試使用 curl..."
                curl -L --progress-bar -o "$SDNET_ZIP" "$SDNET_URL"
            }
        else
            curl -L --progress-bar -o "$SDNET_ZIP" "$SDNET_URL"
        fi

        # 驗證下載是否為有效 ZIP
        if ! unzip -t "$SDNET_ZIP" &>/dev/null; then
            error "下載的檔案不是有效 ZIP，可能是 USU 伺服器問題"
            error "請改用 Kaggle 下載（需免費帳號）："
            error "  pip install kaggle"
            error "  kaggle datasets download -d aniruddhsharma/structural-defects-network-concrete-crack-images"
            error "  mv structural-defects-network-concrete-crack-images.zip $SDNET_ZIP"
            rm -f "$SDNET_ZIP"
            exit 1
        fi
        info "SDNET2018 下載完成：$SDNET_ZIP"
    fi

    info "正在解壓 SDNET2018（約需 1~2 分鐘）..."
    unzip -q "$SDNET_ZIP" -d "$SDNET_DIR"
    info "解壓完成：$SDNET_DIR"

    # 解壓後確認結構（部分版本有子資料夾）
    if [ -d "$SDNET_DIR/SDNET2018" ]; then
        mv "$SDNET_DIR/SDNET2018/"* "$SDNET_DIR/" 2>/dev/null || true
        rmdir "$SDNET_DIR/SDNET2018" 2>/dev/null || true
    fi
fi

# 顯示 SDNET2018 統計
info "SDNET2018 子集統計："
for subset in D W P; do
    for split in C U; do
        dir_path="$SDNET_DIR/${subset}/${split}${subset}"
        if [ -d "$dir_path" ]; then
            count=$(find "$dir_path" -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
            label="crack"
            [ "$split" = "U" ] && label="uncracked"
            echo "  - ${subset}/${split}${subset} (${label}): $count 張"
        fi
    done
done

# =============================================================================
# 3. Roboflow 建築缺陷資料集（mold/stain/peeling 補充來源）
# =============================================================================
section "Roboflow 建築缺陷資料集（mold/stain/peeling 補充）"

ROBOFLOW_DIR="$RAW_DIR/roboflow"
mkdir -p "$ROBOFLOW_DIR"

warn "Roboflow 資料集下載需要免費帳號取得 API Key"
echo ""
echo "  以下資料集含有 mold / stain / peeling 等難以收集的類別："
echo ""
echo "  ① Building Defect on Walls（推薦）"
echo "     https://universe.roboflow.com/builddef2/building-defect-on-walls"
echo "     類別：crack, mold, stain, damp, peeling paint, water seepage"
echo ""
echo "  ② Water Damage Finder"
echo "     https://universe.roboflow.com/jonathan-lewis-8tb5o/water-damage-finder"
echo "     類別：water damage（可對應 stain/mold）"
echo ""
echo "  下載步驟："
echo "  1. 前往 https://roboflow.com 免費註冊（不需信用卡）"
echo "  2. 取得 API Key：https://app.roboflow.com/settings/api"
echo "  3. 安裝 Roboflow CLI："
echo "       pip install roboflow"
echo "  4. 執行下載："
echo "       python scripts/07_download_roboflow.py --api-key YOUR_API_KEY"
echo ""
info "腳本 07_download_roboflow.py 已提供，填入 API Key 後即可執行"

# =============================================================================
# 4. DAGM 2007（說明手動下載方式）
# =============================================================================
section "DAGM 2007（工業表面缺陷）— 手動下載說明"

DAGM_DIR="$RAW_DIR/dagm"

if [ "$(ls -A "$DAGM_DIR" 2>/dev/null)" ]; then
    info "DAGM 資料夾已有內容，略過說明。"
else
    warn "DAGM 2007 需手動下載（建議從 Zenodo，免登入，無需帳號）"
    echo ""
    echo "  方法一（推薦）：zenodo_get 自動下載全部"
    echo "    pip install zenodo-get"
    echo "    cd $DAGM_DIR"
    echo "    zenodo_get 8086136"
    echo ""
    echo "  方法二：wget 批次下載 Class1~6（開發集，約 1.5 GB）"
    echo "    cd $DAGM_DIR"
    cat << 'WGET_EXAMPLE'
    for i in $(seq 1 6); do
        wget -q --show-progress \
            "https://zenodo.org/records/8086136/files/Class${i}.zip" \
            -O "Class${i}.zip" && unzip -q "Class${i}.zip"
    done
WGET_EXAMPLE
    echo ""
    echo "  解壓後結構應為："
    echo "    dagm/Class1/Train/{0/,1/} + Label/"
    echo "    dagm/Class2/Train/{0/,1/} + Label/"
    echo "    ...以此類推到 Class6（或 Class10）"
    echo ""
    info "下載完成後執行：python scripts/04_organize_dagm.py"
fi

# =============================================================================
# 完成摘要
# =============================================================================
section "下載完成摘要"
echo ""
echo "  資料存放位置："
echo "    BD3 sample → $RAW_DIR/bd3/         （僅 37 張，作視覺參考用）"
echo "    SDNET2018  → $RAW_DIR/sdnet2018/   （56,000 張，crack + normal 主要來源）"
echo "    DAGM       → $RAW_DIR/dagm/        （需手動從 Zenodo 下載）"
echo ""
echo "  各類別資料來源規劃："
echo "    normal  → SDNET2018 (充足)"
echo "    crack   → SDNET2018 + DAGM (充足)"
echo "    stain   → DAGM + 台灣自拍（需補充）"
echo "    mold    → 台灣自拍（無公開資料集，需自行收集）"
echo "    peeling → 台灣自拍 + Roboflow（需補充）"
echo "    worn    → 台灣自拍（無公開資料集，需自行收集）"
echo ""
echo "  下一步（依序執行）："
echo "    python scripts/03_organize_sdnet.py  # 整理 SDNET2018（crack + normal）"
echo "    python scripts/04_organize_dagm.py   # 整理 DAGM（需先手動下載）"
echo "    python scripts/05_extract_patches.py # 整理台灣自拍照"
echo "    python scripts/06_verify_dataset.py  # 驗證整體分布並取得 class_weights"
echo ""
echo "  ⚠ mold 和 worn 完全依賴台灣自拍，請優先收集這兩類照片"
echo ""
