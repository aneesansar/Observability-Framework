#!/bin/bash

# Cleanup script for speedygpt-observe-backend
# Removes all __pycache__ directories, .pyc files, and unused legacy files

echo "🧹 Starting cleanup..."

# Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove all .pyc files
echo "Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove all .pyo files
echo "Removing .pyo files..."
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove .DS_Store files (macOS)
echo "Removing .DS_Store files..."
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Remove legacy backup files
echo "Removing legacy backup files..."
rm -f app/main_legacy.py.bak 2>/dev/null || true

# Remove unused files
echo "Removing unused files..."
rm -f app/constants.py 2>/dev/null || true
rm -f app/tenants.py 2>/dev/null || true
rm -f app/evaluate_llm.py 2>/dev/null || true
rm -f app/evaluate_rag.py 2>/dev/null || true
rm -f app/mongo_ingest.py 2>/dev/null || true

# Remove empty services directory
echo "Removing empty services directory..."
rm -rf app/services 2>/dev/null || true

# Remove pytest cache
echo "Removing pytest cache..."
rm -rf .pytest_cache 2>/dev/null || true

# Remove coverage files
echo "Removing coverage files..."
rm -f .coverage 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "To prevent __pycache__ creation in the future:"
echo "1. Set PYTHONDONTWRITEBYTECODE=1 in your environment"
echo "2. Or run: export PYTHONDONTWRITEBYTECODE=1"
echo "3. Or add it to your .env file"
