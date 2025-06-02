#!/bin/bash
# setup.sh - 自動設置 text-vectorify 開發環境

set -e

echo "🚀 設置 text-vectorify 開發環境..."

# 檢查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 檢測到 Python 版本: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ 錯誤: 需要 Python 3.8 或更高版本"
    exit 1
fi

# 創建虛擬環境
if [ ! -d "venv" ]; then
    echo "📦 創建虛擬環境..."
    python3 -m venv venv
else
    echo "📦 虛擬環境已存在"
fi

# 啟動虛擬環境
echo "🔄 啟動虛擬環境..."
source venv/bin/activate

# 升級 pip
echo "⬆️ 升級 pip..."
pip install --upgrade pip

# 選擇安裝類型
echo ""
echo "請選擇安裝類型:"
echo "1) 核心套件 (Core only)"
echo "2) 開發環境 (Development)"
echo "3) 完整安裝 (All dependencies)"
echo -n "請輸入選項 (1-3): "
read choice

case $choice in
    1)
        echo "📦 安裝核心套件..."
        pip install -e .
        ;;
    2)
        echo "📦 安裝開發環境..."
        pip install -e ".[dev]"
        ;;
    3)
        echo "📦 安裝完整套件..."
        pip install -e ".[all,dev]"
        ;;
    *)
        echo "❌ 無效選項，預設安裝核心套件..."
        pip install -e .
        ;;
esac

echo ""
echo "✅ 安裝完成！"
echo ""
echo "下一步："
echo "1. 啟動虛擬環境: source venv/bin/activate"
echo "2. 執行測試: python -m pytest tests/ -v"
echo "3. 查看幫助: text-vectorify --help"
echo ""
echo "範例命令:"
echo "text-vectorify --input examples/sample_input.jsonl --input-field-main \"title\" --process-method \"SentenceBertEmbedder\" --process-model-name \"paraphrase-multilingual-MiniLM-L12-v2\""
