# 開發工具說明

**狀態：** 所有工具已整合到主程式

## 🎉 工具整合完成

原本的開發工具已全部整合到主程式 `text_vectorify.main` 中，提供統一的用戶體驗。

### ✅ 已整合的功能

#### 原 `cache_tool.py` → 主程式cache管理
```bash
# 舊方式
python tools/cache_tool.py --stats
python tools/cache_tool.py --list
python tools/cache_tool.py --clear-all

# 新方式 - 統一接口
python -m text_vectorify.main --show-cache-stats
python -m text_vectorify.main --list-cache-files  
python -m text_vectorify.main --clear-all-caches
```

#### 原 `demo_features.py` → 主程式demo模式
```bash
# 舊方式
python tools/demo_features.py

# 新方式 - 統一接口
python -m text_vectorify.main --demo
```

### 🎯 整合的優勢

1. **簡化學習** - 用戶只需學習一個命令
2. **一致體驗** - 統一的參數和輸出格式
3. **減少複雜度** - 不需要額外的工具腳本
4. **更好的維護性** - 集中的功能和文檔

### 📋 主程式完整功能

```bash
# 查看所有功能
python -m text_vectorify.main --help

# 核心功能
python -m text_vectorify.main --input data.jsonl --input-field-main title --process-method BGEEmbedder

# Cache管理
python -m text_vectorify.main --show-cache-stats
python -m text_vectorify.main --list-cache-files
python -m text_vectorify.main --clear-all-caches

# 學習和演示（智能檔案管理）
python -m text_vectorify.main --demo
```

### 💡 Demo檔案智能管理

Demo功能現在具有智能檔案管理：

- **首次使用**：在當前目錄創建 `demo_data.jsonl`
- **檔案已存在**：詢問用戶是否創建新檔案
  - 選擇 "N"：使用現有檔案
  - 選擇 "Y"：在 `/tmp` 創建臨時檔案
- **自動清理**：臨時檔案由系統自動清理
- **Git忽略**：已加入 `.gitignore` 避免意外提交

## 📁 目錄狀態

`tools/` 目錄現在是空的，所有功能都已整合到主程式中。這個設計符合：

- ✅ Python包裝最佳實踐
- ✅ 用戶體驗優化
- ✅ 單一責任原則
- ✅ 簡化的API設計

## 🔄 遷移指南

如果你之前使用獨立工具，請更新到新的統一命令：

| 舊命令 | 新命令 |
|--------|--------|
| `python tools/cache_tool.py --stats` | `python -m text_vectorify.main --show-cache-stats` |
| `python tools/cache_tool.py --list` | `python -m text_vectorify.main --list-cache-files` |
| `python tools/cache_tool.py --clear-all` | `python -m text_vectorify.main --clear-all-caches` |
| `python tools/demo_features.py` | `python -m text_vectorify.main --demo` |
