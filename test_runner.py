#!/usr/bin/env python3
"""
Simple pytest wrapper for text-vectorify testing
Provides common testing commands for the project
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if optional dependencies are available"""
    dependencies = {
        'pytest': 'Required for testing',
        'pytest-cov': 'Required for coverage',
        'numpy': 'Required core dependency',
        'sentence_transformers': 'Optional - for SentenceBERT embedders',
        'openai': 'Optional - for OpenAI embeddings',
        'torch': 'Optional - for local models',
    }
    
    print("🔍 Checking dependencies:")
    print("-" * 40)
    
    available = {}
    for package, description in dependencies.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package:<20} - {description}")
            available[package] = True
        except ImportError:
            print(f"❌ {package:<20} - {description}")
            available[package] = False
    
    return available


def run_pytest_command(args):
    """Run pytest with given arguments"""
    cmd = ['python', '-m', 'pytest'] + args
    print(f"🚀 Running: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description='Text-Vectorify Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --quick        # Quick smoke tests
  python test_runner.py --core         # Core functionality tests  
  python test_runner.py --integration  # Integration tests
  python test_runner.py --all          # All tests
  python test_runner.py --coverage     # Tests with coverage
  python test_runner.py --deps         # Check dependencies
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick smoke tests')
    parser.add_argument('--core', action='store_true',
                       help='Run core functionality tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage report')
    parser.add_argument('--deps', action='store_true',
                       help='Check dependencies')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        os.chdir(project_root)
        
        if args.deps:
            deps = check_dependencies()
            return 0 if all(deps[pkg] for pkg in ['pytest', 'pytest-cov', 'numpy']) else 1
        
        # Build pytest arguments
        pytest_args = []
        
        if args.verbose:
            pytest_args.append('-v')
        
        if args.quick:
            pytest_args.extend(['-m', 'quick or smoke'])
        elif args.core:
            pytest_args.extend(['-m', 'core'])
        elif args.integration:
            pytest_args.extend(['-m', 'integration'])
        elif args.coverage:
            pytest_args.extend([
                '--cov=text_vectorify',
                '--cov-report=html',
                '--cov-report=term-missing'
            ])
        elif args.all or not any([args.quick, args.core, args.integration, args.coverage]):
            # Default: run all tests
            pass
        
        pytest_args.append('tests/')
        
        # Run tests
        success = run_pytest_command(pytest_args)
        
        if success:
            print("\n🎉 Tests completed successfully!")
        else:
            print("\n❌ Some tests failed!")
        
        return 0 if success else 1
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    import os
    sys.exit(main())
