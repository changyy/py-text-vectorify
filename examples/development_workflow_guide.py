#!/usr/bin/env python3
"""
Text Vectorify Development Workflow Guide

This guide demonstrates the recommended development workflow:
1. CLI testing with small datasets → Cache building
2. Library integration using existing cache → No recomputation

This approach maximizes development efficiency, observability, and testability.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text_vectorify import TextVectorify, EmbedderFactory


class DevelopmentWorkflowDemo:
    """Demonstrates the CLI-first, Library-second development workflow."""
    
    def __init__(self, cache_base_dir: str = "./dev_workflow_cache"):
        self.cache_base_dir = Path(cache_base_dir)
        self.cache_base_dir.mkdir(exist_ok=True)
        
    def step_1_create_test_data(self) -> Path:
        """Step 1: Create small test dataset for CLI experimentation."""
        print("=" * 80)
        print("🔬 STEP 1: Create Small Test Dataset")
        print("=" * 80)
        
        # Create small sample data for initial testing
        test_data = [
            {
                "id": 1,
                "title": "Machine Learning Basics",
                "content": "Introduction to supervised and unsupervised learning algorithms.",
                "category": "AI",
                "author": "Data Scientist"
            },
            {
                "id": 2,
                "title": "深度學習原理",
                "content": "神經網路的基本概念與應用，包含反向傳播演算法。",
                "category": "AI",
                "author": "AI研究員"
            },
            {
                "id": 3,
                "title": "Natural Language Processing",
                "content": "Text processing, tokenization, and semantic analysis techniques.",
                "category": "NLP",
                "author": "NLP Engineer"
            }
        ]
        
        # Save test data
        test_file = self.cache_base_dir / "small_test_data.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Created small test dataset: {test_file}")
        print(f"📊 Dataset size: {len(test_data)} items")
        print("💡 Purpose: Quick CLI testing and cache initialization")
        
        return test_file
    
    def step_2_cli_experimentation(self, test_file: Path):
        """Step 2: Use CLI for quick experimentation and cache building."""
        print("\n" + "=" * 80)
        print("⚡ STEP 2: CLI Experimentation & Cache Building")
        print("=" * 80)
        
        # Define CLI commands for different models
        cli_commands = [
            {
                "model": "BGEEmbedder",
                "description": "Fast, Chinese-optimized model",
                "command": f"""python -m text_vectorify.main \\
  --input {test_file} \\
  --input-field-main title \\
  --input-field-subtitle content \\
  --process-method BGEEmbedder \\
  --cache-dir {self.cache_base_dir}/bge_cache \\
  --output {self.cache_base_dir}/test_bge_output.jsonl"""
            },
            {
                "model": "SentenceBertEmbedder", 
                "description": "Multilingual support",
                "command": f"""python -m text_vectorify.main \\
  --input {test_file} \\
  --input-field-main title \\
  --input-field-subtitle content \\
  --process-method SentenceBertEmbedder \\
  --cache-dir {self.cache_base_dir}/sbert_cache \\
  --output {self.cache_base_dir}/test_sbert_output.jsonl"""
            }
        ]
        
        print("🚀 Recommended CLI commands for initial testing:")
        print("="*50)
        
        for i, cmd_info in enumerate(cli_commands, 1):
            print(f"\n{i}. {cmd_info['model']} ({cmd_info['description']}):")
            print("-" * 60)
            print(cmd_info['command'])
            print()
        
        print("💡 Benefits of CLI-first approach:")
        print("  ✅ Quick iteration and result verification")
        print("  ✅ Cache building for future library use")
        print("  ✅ Parameter tuning without code changes")
        print("  ✅ Immediate output file inspection")
        
        print("\n🔍 After running CLI commands, check:")
        print(f"  📁 Cache directories: {self.cache_base_dir}/*/")
        print(f"  📄 Output files: {self.cache_base_dir}/*_output.jsonl")
        print("  📊 Cache stats: python -m text_vectorify.main --show-cache-stats")
        
    def step_3_library_integration(self, test_file: Path):
        """Step 3: Library integration leveraging existing cache."""
        print("\n" + "=" * 80)
        print("🔗 STEP 3: Library Integration with Cache Reuse")
        print("=" * 80)
        
        # Read test data for library processing
        with open(test_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        print("📚 Library integration example:")
        print("="*50)
        
        print("""
# Example library code that reuses CLI-built cache:

from text_vectorify import EmbedderFactory

def process_with_cache_reuse(data_items: List[Dict]) -> List[Dict]:
    \"\"\"Process data using existing cache from CLI experiments.\"\"\"
    
    # Use same cache directory as CLI
    embedder = EmbedderFactory.create_embedder(
        "BGEEmbedder",
        cache_dir="./dev_workflow_cache/bge_cache"  # Reuse CLI cache!
    )
    
    results = []
    for item in data_items:
        # Combine text fields (same as CLI)
        text = f"{item['title']} {item['content']}"
        
        # This will hit cache from CLI runs - no recomputation!
        vector = embedder.encode(text)
        
        # Add to result
        result = item.copy()
        result['embedding'] = vector
        results.append(result)
    
    return results

# Process data (will reuse cache)
processed_data = process_with_cache_reuse(your_data)
        """)
        
        # Demonstrate actual library usage
        try:
            # Check if we have any cached embedders
            bge_cache_dir = self.cache_base_dir / "bge_cache"
            if bge_cache_dir.exists():
                print("✅ Found existing BGE cache - demonstrating cache reuse...")
                
                embedder = EmbedderFactory.create_embedder(
                    "BGEEmbedder",
                    cache_dir=str(bge_cache_dir)
                )
                
                # Process first item
                item = data[0]
                text = f"{item['title']} {item['content']}"
                vector = embedder.encode(text)
                
                print(f"🎯 Processed item '{item['title']}' using cached embeddings")
                print(f"📏 Vector dimension: {len(vector)}")
                print("⚡ No recomputation needed - used existing cache!")
            else:
                print("ℹ️  No cache found - run CLI commands first to build cache")
                
        except Exception as e:
            print(f"⚠️  Library demo skipped: {e}")
        
        print("\n💡 Key advantages of this workflow:")
        print("  🚀 Fast iteration: CLI for quick tests")
        print("  💾 Cache reuse: No duplicate computations")
        print("  🔍 Observability: Easy result inspection")
        print("  🧪 Testability: Small data → scale up")
        print("  🔄 Seamless transition: CLI → Library")
        
    def step_4_production_pipeline(self):
        """Step 4: Production pipeline template using the workflow."""
        print("\n" + "=" * 80)
        print("🏭 STEP 4: Production Pipeline Template")
        print("=" * 80)
        
        pipeline_template = '''
class ProductionEmbeddingPipeline:
    """Production pipeline leveraging development workflow."""
    
    def __init__(self, cache_dir: str = "./production_cache"):
        self.cache_dir = Path(cache_dir)
        
    def develop_and_test(self, small_sample: List[Dict]) -> Dict:
        """Development phase: CLI testing with small samples."""
        
        # 1. Save small sample for CLI testing
        test_file = self.cache_dir / "dev_sample.jsonl"
        with open(test_file, 'w') as f:
            for item in small_sample:
                f.write(json.dumps(item) + '\\n')
        
        # 2. Return CLI commands for testing
        return {
            "cli_commands": [
                f"python -m text_vectorify.main --input {test_file} "
                f"--input-field-main title --process-method BGEEmbedder "
                f"--cache-dir {self.cache_dir}/bge",
                
                f"python -m text_vectorify.main --input {test_file} "
                f"--input-field-main title --process-method OpenAIEmbedder "
                f"--cache-dir {self.cache_dir}/openai --process-extra-data $OPENAI_API_KEY"
            ],
            "next_step": "Run these commands, then call production_process()"
        }
    
    def production_process(self, full_dataset: List[Dict], 
                          embedder_type: str = "BGEEmbedder") -> List[Dict]:
        """Production phase: Library processing with cache reuse."""
        
        # Reuse cache from development phase
        cache_subdir = self.cache_dir / embedder_type.lower().replace('embedder', '')
        
        embedder = EmbedderFactory.create_embedder(
            embedder_type,
            cache_dir=str(cache_subdir)
        )
        
        results = []
        for item in full_dataset:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            vector = embedder.encode(text)  # Uses cache when possible
            
            result = item.copy()
            result['embedding'] = vector
            results.append(result)
        
        return results

# Usage:
# 1. pipeline.develop_and_test(small_sample) → Get CLI commands
# 2. Run CLI commands → Build cache + verify results  
# 3. pipeline.production_process(full_data) → Reuse cache
        '''
        
        print("🔧 Production Pipeline Template:")
        print("="*50)
        print(pipeline_template)
        
    def step_5_workflow_summary(self):
        """Step 5: Workflow summary and best practices."""
        print("\n" + "=" * 80)
        print("📋 DEVELOPMENT WORKFLOW SUMMARY")
        print("=" * 80)
        
        workflow_steps = [
            ("🔬", "Small Data Creation", "Create 5-10 representative samples"),
            ("⚡", "CLI Experimentation", "Quick model testing & cache building"),
            ("🔍", "Result Verification", "Inspect outputs, tune parameters"),
            ("🔗", "Library Integration", "Code with cache reuse"),
            ("🏭", "Production Scale", "Process full dataset efficiently")
        ]
        
        print("🎯 Recommended Workflow:")
        for emoji, step, description in workflow_steps:
            print(f"  {emoji} {step:20} → {description}")
        
        print("\n💡 Why this workflow works:")
        print("  ✅ Fast feedback loop (CLI)")
        print("  ✅ No wasted computation (cache)")  
        print("  ✅ Easy debugging (small data)")
        print("  ✅ Smooth scaling (library)")
        print("  ✅ Parameter optimization (CLI iteration)")
        
        print("\n🛠️  Key commands to remember:")
        print("  📊 Check cache: python -m text_vectorify.main --show-cache-stats")
        print("  📁 List cache: python -m text_vectorify.main --list-cache-files")
        print("  🗑️  Clear cache: python -m text_vectorify.main --clear-all-caches")
        print("  🎭 Demo mode: python -m text_vectorify.main --demo")
        
        print(f"\n📁 All demo files created in: {self.cache_base_dir}")


def main():
    """Run the complete development workflow demonstration."""
    print("🚀 TEXT VECTORIFY DEVELOPMENT WORKFLOW GUIDE")
    print("Best practices for efficient development with CLI-first approach")
    
    # Initialize workflow demo
    demo = DevelopmentWorkflowDemo()
    
    # Run all workflow steps
    test_file = demo.step_1_create_test_data()
    demo.step_2_cli_experimentation(test_file)
    demo.step_3_library_integration(test_file)
    demo.step_4_production_pipeline()
    demo.step_5_workflow_summary()
    
    print("\n" + "=" * 80)
    print("🎉 WORKFLOW GUIDE COMPLETE!")
    print("Follow these steps for efficient text vectorization development")
    print("=" * 80)


if __name__ == "__main__":
    main()
