# Dataset Description

This document describes all datasets used in the AlgoGen research project.

## Raw Datasets

### SVL Dataset (`raw_datasets/svl_dataset/`)
**Description**: Original SVL (Synthetic Visual Language) dataset containing algorithm generation tasks.

**Structure**:
- `train/` - Training data
- `val/` - Validation data  
- `test/` - Test data
- `lcs/` - Longest Common Subsequence problems
- `dijkstra/` - Dijkstra algorithm problems
- `long/` - Long sequence problems
- `all_test/` - All test problems
- `small_test/` - Small test problems
- `mid_test/` - Medium test problems

**Format**: JSON files containing algorithm problems with visual representations.

### SVL Dataset No Semantics (`raw_datasets/svl_dataset_no_semantics/`)
**Description**: SVL dataset with semantic information removed for ablation studies.

**Structure**: Similar to original SVL dataset but without semantic annotations.

## Processed Datasets

### Processed Datasets (`processed_datasets/`)
**Description**: Optimized datasets for model training and evaluation.

**Content**:
- Various JSONL files for different models and algorithms
- Optimized for training efficiency
- Standardized format for evaluation

**Format**: JSONL format with each line containing a JSON object with input/output pairs.

## Evaluation Results

### Model Evaluation Reports (`evaluation_results/`)
**Description**: JSON files containing evaluation results for different models and configurations.

**Content**:
- Model evaluation reports in JSON format
- Detailed analysis results
- Confidence interval calculations
- Soft matching metrics

### Detailed Results (`evaluation_results/`)
**Description**: Comprehensive evaluation results and analysis.

**Content**:
- Detailed evaluation results
- Summary statistics
- FFI (Function-Function Intersection) results
- Various analysis outputs

### Visualization Assets (`evaluation_results/`)
**Description**: Visual outputs and analysis results.

**Content**:
- Algorithm animations
- Visualization assets
- Analysis charts and graphs

## Data Format Specifications

### JSONL Format
Each line in JSONL files contains a JSON object with the following structure:
```json
{
  "input": "Problem description or input",
  "output": "Expected solution or output",
  "metadata": {
    "algorithm_type": "lcs|dijkstra|sort|...",
    "difficulty": "easy|medium|hard",
    "length": 123
  }
}
```

### Evaluation Report Format
Evaluation reports contain metrics and results:
```json
{
  "model_name": "model_name",
  "dataset": "dataset_name",
  "metrics": {
    "exact_match": 0.85,
    "soft_match": 0.92,
    "ffi_score": 0.78
  },
  "detailed_results": [...]
}
```

## Usage Notes

1. **Training**: Use processed datasets for model training
2. **Evaluation**: Use evaluation results for analysis and comparison
3. **Reproduction**: All datasets are provided to ensure experiment reproducibility
4. **Format Conversion**: Raw datasets can be converted to processed formats using provided scripts

## Citation
When using these datasets, please cite the original paper and this repository. 