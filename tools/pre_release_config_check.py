#!/usr/bin/env python3
"""
發佈前配置文件快速驗證 / Pre-release Configuration Quick Validation

此腳本用於發佈前快速檢查配置文件的基本完整性
This script is used for quick pre-release validation of configuration file integrity
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def quick_validate_configs() -> bool:
    """快速驗證配置文件"""
    print("🚀 發佈前配置文件快速驗證...")
    
    configs_dir = project_root / "configs"
    json_files = list(configs_dir.glob("*.json"))
    
    if not json_files:
        print("❌ 未找到配置文件")
        return False
    
    errors = []
    warnings = []
    
    for config_file in sorted(json_files):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 基本檢查
            file_name = config_file.name
            
            # 檢查是否有 embedder_type (新格式)
            if "embedder_type" in config:
                embedder_type = config["embedder_type"]
                
                if embedder_type == "TFIDFEmbedder":
                    # 檢查 TF-IDF 配置
                    config_data = config.get("config", {})
                    tokenizer = config_data.get("chinese_tokenizer")
                    
                    if tokenizer not in ["spacy", "jieba", "pkuseg"]:
                        errors.append(f"{file_name}: 無效的分詞器 '{tokenizer}'")
                    
                    if "ngram_range" in config_data:
                        ngram_range = config_data["ngram_range"]
                        if not isinstance(ngram_range, list) or len(ngram_range) != 2:
                            errors.append(f"{file_name}: ngram_range 格式錯誤")
                
                elif embedder_type == "MultiLayerEmbedder":
                    # 檢查多層配置
                    config_data = config.get("config", {})
                    embedder_configs = config_data.get("embedder_configs", [])
                    
                    if not embedder_configs:
                        errors.append(f"{file_name}: 多層配置缺少 embedder_configs")
                    
                    for i, layer in enumerate(embedder_configs):
                        if "type" not in layer:
                            errors.append(f"{file_name}: 層 {i} 缺少 type")
                        if "config" not in layer:
                            errors.append(f"{file_name}: 層 {i} 缺少 config")
                
            elif "layers" in config:
                # 遺留格式
                warnings.append(f"{file_name}: 使用遺留格式，建議升級")
                
            else:
                errors.append(f"{file_name}: 無法識別的配置格式")
            
            print(f"✅ {file_name}")
            
        except json.JSONDecodeError as e:
            errors.append(f"{config_file.name}: JSON 語法錯誤 - {e}")
            print(f"❌ {config_file.name}")
        except Exception as e:
            errors.append(f"{config_file.name}: 驗證錯誤 - {e}")
            print(f"❌ {config_file.name}")
    
    # 輸出結果
    print(f"\n📊 驗證結果: {len(json_files)} 個配置文件")
    
    if warnings:
        print(f"\n⚠️  警告 ({len(warnings)}):")
        for warning in warnings:
            print(f"   {warning}")
    
    if errors:
        print(f"\n❌ 錯誤 ({len(errors)}):")
        for error in errors:
            print(f"   {error}")
        return False
    
    print("\n🎉 所有配置文件驗證通過！")
    return True


def check_required_configs() -> bool:
    """檢查必需的配置文件是否存在"""
    configs_dir = project_root / "configs"
    
    required_configs = [
        "tfidf_spacy_example.json",
        "tfidf_jieba_example.json",
        "multi_layer_simple.json",
        "multi_layer_spacy_recommended.json"
    ]
    
    missing = []
    for config_name in required_configs:
        config_path = configs_dir / config_name
        if not config_path.exists():
            missing.append(config_name)
    
    if missing:
        print(f"❌ 缺少必需的配置文件: {', '.join(missing)}")
        return False
    
    print("✅ 所有必需的配置文件都存在")
    return True


def check_documentation() -> bool:
    """檢查文檔是否存在"""
    readme_path = project_root / "configs" / "README.md"
    
    if not readme_path.exists():
        print("❌ 缺少 configs/README.md 文檔")
        return False
    
    print("✅ 配置文檔存在")
    return True


def main():
    """主函數"""
    print("🔧 py-text-vectorify 發佈前配置驗證")
    print("=" * 50)
    
    all_passed = True
    
    # 檢查必需配置文件
    if not check_required_configs():
        all_passed = False
    
    # 檢查文檔
    if not check_documentation():
        all_passed = False
    
    # 快速驗證配置
    if not quick_validate_configs():
        all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 發佈前驗證通過！所有配置文件準備就緒。")
        sys.exit(0)
    else:
        print("❌ 發佈前驗證失敗！請修復上述問題。")
        sys.exit(1)


if __name__ == "__main__":
    main()
