#!/usr/bin/env python3
"""
CLI-First Development Workflow Demo

This script demonstrates the recommended CLI-first development approach:
1. Start with CLI for quick testing and cache building
2. Switch to library for integration while reusing cache
3. Benefit from zero recomputation

Note: This demo simulates the workflow without requiring model dependencies.
"""

import json
import subprocess
import sys
from pathlib import Path

def create_demo_data():
    """Create sample data for demonstration."""
    print("🔬 STEP 1: Creating Small Test Dataset")
    print("=" * 60)
    
    demo_data = [
        {
            "id": 1,
            "title": "Python Data Science",
            "content": "Learn pandas, numpy, and matplotlib for data analysis",
            "category": "Programming"
        },
        {
            "id": 2,  
            "title": "機器學習入門",
            "content": "深度學習與神經網路的基礎概念與實作",
            "category": "AI"
        },
        {
            "id": 3,
            "title": "Web Development",
            "content": "Building modern web applications with React and Node.js",
            "category": "Development"
        }
    ]
    
    demo_file = Path("workflow_demo_data.jsonl")
    with open(demo_file, 'w', encoding='utf-8') as f:
        for item in demo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Created demo dataset: {demo_file}")
    print(f"📊 Dataset size: {len(demo_data)} items")
    print("💡 Purpose: CLI testing and cache initialization\n")
    
    return demo_file

def show_cli_commands(demo_file):
    """Show CLI commands for different scenarios."""
    print("⚡ STEP 2: CLI Experimentation Commands")
    print("=" * 60)
    
    commands = [
        {
            "name": "BGE Embedder (Chinese Optimized)",
            "description": "Fast and effective for Chinese/English text",
            "command": f"""python -m text_vectorify.main \\
  --input {demo_file} \\
  --input-field-main title \\
  --input-field-subtitle content \\
  --process-method BGEEmbedder \\
  --cache-dir ./workflow_cache/bge \\
  --output ./workflow_results_bge.jsonl"""
        },
        {
            "name": "SentenceBERT (Multilingual)",
            "description": "Good for multilingual scenarios", 
            "command": f"""python -m text_vectorify.main \\
  --input {demo_file} \\
  --input-field-main title \\
  --input-field-subtitle content \\
  --process-method SentenceBertEmbedder \\
  --cache-dir ./workflow_cache/sbert \\
  --output ./workflow_results_sbert.jsonl"""
        },
        {
            "name": "OpenAI (Premium Quality)",
            "description": "Highest quality, requires API key",
            "command": f"""python -m text_vectorify.main \\
  --input {demo_file} \\
  --input-field-main title \\
  --input-field-subtitle content \\
  --process-method OpenAIEmbedder \\
  --model-name text-embedding-3-small \\
  --extra-data $OPENAI_API_KEY \\
  --cache-dir ./workflow_cache/openai \\
  --output ./workflow_results_openai.jsonl"""
        }
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd['name']}")
        print(f"   Description: {cmd['description']}")
        print(f"   Command:")
        print(f"   {cmd['command']}")
        print()
    
    print("🔍 After running CLI commands:")
    print("  • Inspect output: head -3 workflow_results_*.jsonl")
    print("  • Check cache: python -m text_vectorify.main --show-cache-stats")
    print("  • List files: python -m text_vectorify.main --list-cache-files")
    print()

def show_library_integration():
    """Show library integration code that reuses CLI cache."""
    print("🔗 STEP 3: Library Integration with Cache Reuse")
    print("=" * 60)
    
    library_code = '''
# Library integration example - reuses CLI-built cache
from text_vectorify import EmbedderFactory
import json

def process_with_cache_reuse(data_list, embedder_type="BGEEmbedder"):
    """Process data using existing cache from CLI experiments."""
    
    # Use same cache directory as CLI - no recomputation!
    cache_mapping = {
        "BGEEmbedder": "./workflow_cache/bge",
        "SentenceBertEmbedder": "./workflow_cache/sbert", 
        "OpenAIEmbedder": "./workflow_cache/openai"
    }
    
    cache_dir = cache_mapping.get(embedder_type, "./workflow_cache/default")
    
    # Create embedder with CLI cache directory
    embedder = EmbedderFactory.create_embedder(
        embedder_type,
        cache_dir=cache_dir  # Reuse CLI cache!
    )
    
    results = []
    for item in data_list:
        # Combine text fields (same as CLI)
        text = f"{item['title']} {item['content']}"
        
        # This will hit cache from CLI runs - instant results!
        vector = embedder.encode(text)
        
        # Add embedding to result
        result = item.copy()
        result['embedding'] = vector
        result['vector_dim'] = len(vector)
        results.append(result)
    
    return results

# Usage example:
# 1. Run CLI commands first (builds cache)
# 2. Then use library (reuses cache)

with open("workflow_demo_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Process with instant cache hits
processed_data = process_with_cache_reuse(data, "BGEEmbedder")

# Results include embeddings from cached computations
for item in processed_data:
    print(f"'{item['title']}' -> {item['vector_dim']}D vector")
    '''
    
    print("📚 Library Code Example:")
    print(library_code)

def show_production_pipeline():
    """Show production pipeline template."""
    print("🏭 STEP 4: Production Pipeline Pattern")
    print("=" * 60)
    
    pipeline_code = '''
class CLIFirstEmbeddingPipeline:
    """Production pipeline using CLI-first development pattern."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.cache_base = f"./pipelines/{project_name}_cache"
        
    def development_phase(self, sample_data: list) -> dict:
        """Phase 1: CLI development with small samples."""
        
        # Save sample for CLI testing
        sample_file = f"{self.project_name}_sample.jsonl"
        with open(sample_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\\n')
        
        # Return CLI commands for testing
        return {
            "sample_file": sample_file,
            "cli_commands": {
                "bge": f"python -m text_vectorify.main --input {sample_file} "
                       f"--input-field-main title --process-method BGEEmbedder "
                       f"--cache-dir {self.cache_base}/bge",
                       
                "openai": f"python -m text_vectorify.main --input {sample_file} "
                         f"--input-field-main title --process-method OpenAIEmbedder "
                         f"--cache-dir {self.cache_base}/openai "
                         f"--extra-data $OPENAI_API_KEY"
            },
            "next_steps": [
                "1. Run CLI commands to test and build cache",
                "2. Inspect output files and verify results", 
                "3. Call production_phase() for full processing"
            ]
        }
    
    def production_phase(self, full_dataset: list, embedder_type: str) -> list:
        """Phase 2: Library processing with cache reuse."""
        
        # Map to CLI cache directories
        cache_dir = f"{self.cache_base}/{embedder_type.lower().replace('embedder', '')}"
        
        embedder = EmbedderFactory.create_embedder(
            embedder_type,
            cache_dir=cache_dir  # Reuse development cache!
        )
        
        results = []
        for item in full_dataset:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            vector = embedder.encode(text)  # Cache hits = instant results
            
            result = item.copy()
            result['embedding'] = vector
            results.append(result)
        
        return results

# Usage workflow:
# 1. pipeline = CLIFirstEmbeddingPipeline("my_project")
# 2. dev_info = pipeline.development_phase(small_sample)
# 3. Run dev_info["cli_commands"] in terminal
# 4. results = pipeline.production_phase(full_data, "BGEEmbedder")
    '''
    
    print("🔧 Production Pipeline Template:")
    print(pipeline_code)

def show_workflow_benefits():
    """Show the benefits of CLI-first approach."""
    print("💡 WORKFLOW BENEFITS SUMMARY")
    print("=" * 60)
    
    benefits = [
        ("🔬", "Fast Iteration", "CLI allows quick parameter testing"),
        ("💾", "Smart Caching", "Zero recomputation between CLI and library"), 
        ("🔍", "Easy Debugging", "Inspect outputs immediately with CLI"),
        ("🧪", "Incremental Testing", "Start small, scale gradually"),
        ("⚡", "Performance", "Cache hits provide instant results"),
        ("🎯", "Workflow Clarity", "Clear development → production path"),
        ("🛠️", "Tool Integration", "Seamless CLI → library transition"),
        ("📊", "Observability", "Easy cache monitoring and management")
    ]
    
    print("🎯 Why CLI-First Works:")
    for emoji, title, description in benefits:
        print(f"  {emoji} {title:20} → {description}")
    
    print("\n🔄 Development Lifecycle:")
    lifecycle = [
        "1. 🔬 Create small test data (5-10 samples)",
        "2. ⚡ Run CLI commands to test models",
        "3. 🔍 Inspect outputs and tune parameters", 
        "4. 🔗 Switch to library with cache reuse",
        "5. 🏭 Scale to production with zero waste"
    ]
    
    for step in lifecycle:
        print(f"     {step}")
    
    print(f"\n📁 Demo files will be created in current directory")
    print("🛠️  Key CLI commands to remember:")
    print("     📊 python -m text_vectorify.main --show-cache-stats")
    print("     📁 python -m text_vectorify.main --list-cache-files") 
    print("     🗑️  python -m text_vectorify.main --clear-all-caches")

def main():
    """Run the complete CLI-first workflow demonstration."""
    print("🚀 CLI-FIRST DEVELOPMENT WORKFLOW DEMO")
    print("Efficient text vectorization development pattern")
    print("=" * 70)
    print()
    
    # Step 1: Create demo data
    demo_file = create_demo_data()
    
    # Step 2: Show CLI commands
    show_cli_commands(demo_file)
    
    # Step 3: Show library integration
    show_library_integration()
    
    # Step 4: Show production pipeline
    show_production_pipeline()
    
    # Step 5: Show workflow benefits
    show_workflow_benefits()
    
    print("\n" + "=" * 70)
    print("🎉 CLI-FIRST WORKFLOW DEMO COMPLETE!")
    print("Follow this pattern for efficient text vectorization development")
    print("=" * 70)

if __name__ == "__main__":
    main()
