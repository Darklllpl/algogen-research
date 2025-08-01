#!/bin/bash

# AlgoGen Project Statistics Script
# This script shows statistics about the current project structure

echo "=== AlgoGen Project Statistics ==="
echo ""

# Count Python files by category
echo "📊 Python Files by Category:"
echo ""

# Research code
research_files=$(find code -name "*.py" | wc -l)
echo "🔬 Research Code: $research_files files"
find code -name "*.py" | sed 's/^/  /'

echo ""

# Application code
app_files=$(find applications -name "*.py" | wc -l)
echo "🖥️  Application Code: $app_files files"
find applications -name "*.py" | sed 's/^/  /'

echo ""

# Total Python files
total_python=$(find . -name "*.py" | wc -l)
echo "📈 Total Python Files: $total_python"

echo ""

# Count files by directory
echo "📁 Files by Directory:"
echo ""

echo "Code directories:"
for dir in code/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.py" | wc -l)
        dirname=$(basename "$dir")
        echo "  $dirname/: $count files"
    fi
done

echo ""

echo "Application directories:"
for dir in applications/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.py" | wc -l)
        dirname=$(basename "$dir")
        echo "  $dirname/: $count files"
    fi
done

echo ""

# Dataset statistics
echo "📊 Dataset Statistics:"
echo ""

if [ -d "datasets" ]; then
    echo "Raw datasets:"
    for dir in datasets/raw_datasets/*/; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -type f | wc -l)
            dirname=$(basename "$dir")
            echo "  $dirname/: $count files"
        fi
    done
    
    echo ""
    echo "Processed datasets:"
    processed_count=$(find datasets/processed_datasets -type f | wc -l)
    echo "  processed_datasets/: $processed_count files"
    
    echo ""
    echo "Evaluation results:"
    eval_count=$(find datasets/evaluation_results -type f | wc -l)
    echo "  evaluation_results/: $eval_count files"
fi

echo ""

# File size statistics
echo "💾 File Size Statistics:"
echo ""

total_size=$(du -sh . | cut -f1)
echo "Total project size: $total_size"

echo ""

# Algorithm coverage in applications
echo "🎯 Algorithm Coverage (AlgoGen v6.0):"
echo ""

if [ -d "applications/algogen_v6" ]; then
    echo "Sorting algorithms:"
    find applications/algogen_v6/sort -name "*_tracker_v5.py" | sed 's/.*\///' | sed 's/_tracker_v5.py//' | sed 's/^/  /'
    
    echo ""
    echo "Graph algorithms:"
    find applications/algogen_v6/graph -name "*_tracker_v5.py" | sed 's/.*\///' | sed 's/_tracker_v5.py//' | sed 's/^/  /'
    
    echo ""
    echo "Dynamic programming:"
    find applications/algogen_v6/dp -name "*_tracker_v5.py" | sed 's/.*\///' | sed 's/_tracker_v5.py//' | sed 's/^/  /'
fi

echo ""
echo "=== End of Statistics ===" 