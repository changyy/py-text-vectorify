# Development Guide

## Quick Start

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
# venv\Scripts\activate
```

### 2. Install Packages

```bash
# Install core package (basic functionality only)
pip install -e .

# Install development version (includes development tools)
pip install -e ".[dev]"

# Install full version (includes all model support)
pip install -e ".[all]"
```

### 3. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests and generate coverage report
python -m pytest tests/ --cov=text_vectorify --cov-report=html

# Run quick test
python quick_test.py
```

### 4. Code Formatting

```bash
# Format code
python -m black text_vectorify tests examples

# Sort imports
python -m isort text_vectorify tests examples

# Check code style
python -m flake8 text_vectorify tests

# Type checking
python -m mypy text_vectorify
```

### 5. Build Package

```bash
# Build source distribution
python setup.py sdist

# Build wheel package
python setup.py bdist_wheel

# Use build tool (recommended)
pip install build
python -m build
```

## Common Commands

| Function | Command |
|----------|---------|
| Create virtual environment | `python3 -m venv venv` |
| Activate virtual environment | `source venv/bin/activate` |
| Install development version | `pip install -e ".[dev]"` |
| Run tests | `python -m pytest tests/ -v` |
| Format code | `python -m black .` |
| Check syntax | `python -m flake8 text_vectorify` |
| Clean build files | `rm -rf build/ dist/ *.egg-info/` |

## Project Structure

```
text-vectorify/
├── setup.py              # Package installation configuration
├── pyproject.toml         # Tool configuration
├── requirements*.txt      # Dependencies
├── README.md             # Project description
├── DEVELOPMENT.md        # Development guide
├── text_vectorify/       # Main source code
│   ├── __init__.py
│   ├── main.py           # CLI entry point
│   ├── vectorify.py      # Main class
│   ├── factory.py        # Factory class
│   └── embedders/        # Embedder modules
├── tests/                # Test files
├── examples/             # Example files
└── docs/                 # Documentation (optional)
```

## Release Process

1. Update version number in `text_vectorify/__init__.py`
2. Update `CHANGELOG.md` (if exists)
3. Run tests to ensure everything works
4. Build package: `python -m build`
5. Publish to PyPI: `python -m twine upload dist/*`

## Notes

- Use Python 3.8+
- Follow PEP 8 code style
- Add tests for new features
- Update documentation
