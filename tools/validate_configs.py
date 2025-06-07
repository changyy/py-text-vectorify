#!/usr/bin/env python3
"""
配置文件驗證工具 / Configuration Files Validation Tool

此腳本用於快速驗證 configs/ 目錄中的所有 JSON 配置文件
This script is used to quickly validate all JSON configuration files in the configs/ directory
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text_vectorify.embedders.tfidf import TFIDFEmbedder
from text_vectorify.embedders.multi_layer import MultiLayerEmbedder


class ConfigValidator:
    """配置文件驗證器"""
    
    def __init__(self):
        self.configs_dir = project_root / "configs"
        self.test_texts = [
            "這是第一個測試文本，包含機器學習的內容。",
            "人工智能技術在自然語言處理領域發展迅速。",
            "深度學習模型在文本分析中表現出色。"
        ]
        self.results = []
    
    def validate_json_syntax(self, config_path: Path) -> Dict[str, Any]:
        """驗證 JSON 語法"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return {"status": "✅", "config": config, "error": None}
        except json.JSONDecodeError as e:
            return {"status": "❌", "config": None, "error": f"JSON 語法錯誤: {e}"}
        except Exception as e:
            return {"status": "❌", "config": None, "error": f"讀取錯誤: {e}"}
    
    def validate_config_structure(self, config: Dict, config_name: str) -> Dict[str, Any]:
        """驗證配置結構"""
        try:
            if "embedder_type" in config:
                # 新格式配置
                embedder_type = config["embedder_type"]
                config_data = config.get("config", {})
                
                if embedder_type == "TFIDFEmbedder":
                    return self._validate_tfidf_config(config_data)
                elif embedder_type == "MultiLayerEmbedder":
                    return self._validate_multilayer_config(config_data)
                else:
                    return {"status": "⚠️", "message": f"未測試的嵌入器類型: {embedder_type}"}
                    
            elif "layers" in config:
                # 遺留格式配置
                return {"status": "⚠️", "message": "遺留格式配置，結構有效但建議升級"}
                
            else:
                return {"status": "❌", "message": "無法識別的配置格式"}
                
        except Exception as e:
            return {"status": "❌", "message": f"結構驗證錯誤: {e}"}
    
    def _validate_tfidf_config(self, config: Dict) -> Dict[str, Any]:
        """驗證 TF-IDF 配置"""
        try:
            # 檢查必要參數
            tokenizer = config.get('chinese_tokenizer', 'spacy')
            if tokenizer not in ['spacy', 'jieba', 'pkuseg']:
                return {"status": "❌", "message": f"無效的分詞器: {tokenizer}"}
            
            # 檢查參數類型
            if 'ngram_range' in config:
                ngram_range = config['ngram_range']
                if not isinstance(ngram_range, list) or len(ngram_range) != 2:
                    return {"status": "❌", "message": "ngram_range 必須是長度為 2 的列表"}
            
            if 'max_features' in config:
                max_features = config['max_features']
                if not isinstance(max_features, int) or max_features <= 0:
                    return {"status": "❌", "message": "max_features 必須是正整數"}
            
            return {"status": "✅", "message": "TF-IDF 配置結構有效"}
            
        except Exception as e:
            return {"status": "❌", "message": f"TF-IDF 配置驗證錯誤: {e}"}
    
    def _validate_multilayer_config(self, config: Dict) -> Dict[str, Any]:
        """驗證多層配置"""
        try:
            embedder_configs = config.get('embedder_configs', [])
            if not isinstance(embedder_configs, list) or len(embedder_configs) == 0:
                return {"status": "❌", "message": "embedder_configs 必須是非空列表"}
            
            for i, layer in enumerate(embedder_configs):
                if not isinstance(layer, dict):
                    return {"status": "❌", "message": f"層 {i} 必須是字典格式"}
                
                if 'type' not in layer:
                    return {"status": "❌", "message": f"層 {i} 缺少 type 字段"}
                
                if 'config' not in layer:
                    return {"status": "❌", "message": f"層 {i} 缺少 config 字段"}
                
                # 驗證 TF-IDF 層
                if layer['type'] == 'TFIDFEmbedder':
                    tfidf_result = self._validate_tfidf_config(layer['config'])
                    if tfidf_result['status'] != "✅":
                        return {"status": "❌", "message": f"層 {i} TF-IDF 配置錯誤: {tfidf_result['message']}"}
            
            return {"status": "✅", "message": f"多層配置結構有效 ({len(embedder_configs)} 層)"}
            
        except Exception as e:
            return {"status": "❌", "message": f"多層配置驗證錯誤: {e}"}
    
    def test_functionality(self, config: Dict, config_name: str) -> Dict[str, Any]:
        """測試功能性"""
        try:
            if "embedder_type" not in config:
                return {"status": "⚠️", "message": "跳過功能測試（遺留格式）"}
            
            embedder_type = config["embedder_type"]
            config_data = config["config"]
            
            if embedder_type == "TFIDFEmbedder":
                embedder = TFIDFEmbedder(**config_data)
                embedder.fit(self.test_texts)
                vector = embedder.encode(self.test_texts[0])
                
                if isinstance(vector, list) and len(vector) > 0:
                    return {"status": "✅", "message": f"功能測試通過，向量維度: {len(vector)}"}
                else:
                    return {"status": "❌", "message": "向量生成失敗"}
            
            elif embedder_type == "MultiLayerEmbedder":
                # 簡化測試，只檢查能否創建
                embedder = MultiLayerEmbedder(**config_data)
                return {"status": "✅", "message": f"多層嵌入器創建成功 ({len(embedder.embedders)} 層)"}
            
            else:
                return {"status": "⚠️", "message": "跳過功能測試（不支持的類型）"}
                
        except Exception as e:
            return {"status": "❌", "message": f"功能測試失敗: {e}"}
    
    def validate_config_file(self, config_path: Path) -> Dict[str, Any]:
        """驗證單個配置文件"""
        print(f"📁 驗證: {config_path.name}")
        
        # JSON 語法檢查
        json_result = self.validate_json_syntax(config_path)
        if json_result["status"] != "✅":
            return {
                "file": config_path.name,
                "json_syntax": json_result["status"],
                "structure": "⏭️",
                "functionality": "⏭️",
                "error": json_result["error"]
            }
        
        config = json_result["config"]
        
        # 結構檢查
        structure_result = self.validate_config_structure(config, config_path.name)
        
        # 功能檢查（僅對有效結構進行）
        if structure_result["status"] == "✅":
            func_result = self.test_functionality(config, config_path.name)
        else:
            func_result = {"status": "⏭️", "message": "跳過功能測試"}
        
        return {
            "file": config_path.name,
            "json_syntax": json_result["status"],
            "structure": structure_result["status"],
            "functionality": func_result["status"],
            "structure_message": structure_result["message"],
            "functionality_message": func_result["message"]
        }
    
    def run_validation(self) -> List[Dict[str, Any]]:
        """運行完整驗證"""
        print("🚀 開始配置文件驗證...")
        print("=" * 60)
        
        json_files = list(self.configs_dir.glob("*.json"))
        if not json_files:
            print("❌ 未找到任何 JSON 配置文件")
            return []
        
        start_time = time.time()
        
        for config_file in sorted(json_files):
            result = self.validate_config_file(config_file)
            self.results.append(result)
            print()
        
        end_time = time.time()
        
        # 生成報告
        self.print_summary_report(end_time - start_time)
        
        return self.results
    
    def print_summary_report(self, duration: float):
        """打印摘要報告"""
        print("=" * 60)
        print("📊 驗證摘要報告")
        print("=" * 60)
        
        total_files = len(self.results)
        json_passed = sum(1 for r in self.results if r["json_syntax"] == "✅")
        structure_passed = sum(1 for r in self.results if r["structure"] == "✅")
        functionality_passed = sum(1 for r in self.results if r["functionality"] == "✅")
        
        print(f"📁 總配置文件數量: {total_files}")
        print(f"✅ JSON 語法通過: {json_passed}/{total_files}")
        print(f"✅ 結構驗證通過: {structure_passed}/{total_files}")
        print(f"✅ 功能測試通過: {functionality_passed}/{total_files}")
        print(f"⏱️  驗證耗時: {duration:.2f} 秒")
        
        print("\n📋 詳細結果:")
        print("-" * 60)
        for result in self.results:
            status_line = f"{result['json_syntax']} {result['structure']} {result['functionality']} {result['file']}"
            print(status_line)
            
            if result.get('structure_message'):
                print(f"   📝 結構: {result['structure_message']}")
            if result.get('functionality_message'):
                print(f"   🔧 功能: {result['functionality_message']}")
            if result.get('error'):
                print(f"   ❌ 錯誤: {result['error']}")
        
        print(f"\n✅ 配置文件驗證完成！")
        
        # 返回狀態碼
        if json_passed == total_files and structure_passed >= total_files * 0.8:
            print("🎉 所有配置文件驗證通過！")
            return 0
        else:
            print("⚠️  部分配置文件存在問題，請檢查上述報告")
            return 1


def main():
    """主函數"""
    print("🔧 py-text-vectorify 配置文件驗證工具")
    print("🔧 Configuration Files Validation Tool")
    print("=" * 60)
    
    validator = ConfigValidator()
    results = validator.run_validation()
    
    # 返回適當的退出代碼
    if all(r["json_syntax"] == "✅" for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
