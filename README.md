# AlgoGen: Algorithm Generation Research Repository

This repository contains the complete code and datasets for the AlgoGen research project, which focuses on algorithm generation using large language models.

## Project Structure

### Research Code (`code/`)

#### 1. Data Processing (`code/data_processing/`)
- `create_gemma_dataset.py` - Generate datasets for Gemma model training
- `create_t5_dataset_snapshot.py` - Create T5 model dataset snapshots
- `create_t5_native_dataset.py` - Generate native T5 datasets

#### 2. Model Training (`code/model_training/`)
- `train_gemma.py` - Train Gemma models
- `train_delta_predictor.py` - Train delta prediction models

#### 3. Inference (`code/inference/`)
- `run_cot_autoregressive_newest.py` - Latest chain-of-thought autoregressive inference
- `run_cot_autoregressive.py` - Basic chain-of-thought autoregressive inference
- `step_by_step_generator.py` - Step-by-step generation
- `step_by_step_generator_gemma.py` - Gemma step-by-step generation
- `inference.py` - General inference utilities
- `gpt.py` - GPT model integration

#### 4. Evaluation (`code/evaluation/`)
- `evaluation.py` - Main evaluation script
- `evaluation_better.py` - Improved evaluation script
- `evaluate_jsonl.py` - JSONL format evaluation
- `evaluate_jsonl_gemma.py` - Gemma model JSONL evaluation
- `em_confidence_interval.py` - Exact match confidence interval calculation
- `calculate_soft_ffi.py` - Soft matching FFI calculation

#### 5. Visualization (`code/visualization/`)
- `renderer.py` - Data rendering utilities

### Applications (`applications/`)

#### AlgoGen v6.0 (`applications/algogen_v6/`)
A complete algorithm visualization application system based on SVL 5.0 specification.

**Core Components**:
- **Web Application** (`web/`): Flask-based web interface with algorithm visualization
- **Algorithm Trackers**: Specialized trackers for different algorithm types
  - **Sorting Algorithms** (`sort/`): Bubble, Heap, Insertion, Merge, Quick, Selection sort
  - **Graph Algorithms** (`graph/`): BFS, DFS, Dijkstra, Bellman-Ford, Kruskal, Prim, Topological sort
  - **Dynamic Programming** (`dp/`): LCS, Knapsack, Edit Distance
- **Rendering System**: Advanced visualization engine with style management
- **Validation Tools**: Data validation and testing utilities

**Key Features**:
- Interactive algorithm visualization
- SVL 5.0 compliant rendering
- LLM integration for creative generation
- Comprehensive algorithm coverage
- Web-based user interface

**Quick Start**:
```bash
cd applications/algogen_v6/web
python app.py
# Visit http://localhost:5000
```

### Dataset Organization

#### 1. Raw Datasets (`datasets/raw_datasets/`)
- `svl_dataset/` - Original SVL datasets with train/val/test splits
- `svl_dataset_no_semantics/` - SVL datasets without semantics

#### 2. Processed Datasets (`datasets/processed_datasets/`)
- Various JSONL files for different models and algorithms
- Optimized for training and evaluation

#### 3. Evaluation Results (`datasets/evaluation_results/`)
- Complete evaluation results from all experiments
- Model evaluation reports in JSON format
- Log files and analysis outputs
- Visualization assets

## Usage

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Quick Start

#### Research Experiments
```bash
# Run all research experiments
./run_experiments.sh
```

#### Application Launch
```bash
# Start AlgoGen v6.0 application
./run_applications.sh v6
```

### Key Features
- **Multi-model Support**: Supports Gemma, T5, GPT, and DeepSeek models
- **Chain-of-Thought**: Implements various CoT inference strategies
- **Comprehensive Evaluation**: Multiple evaluation metrics and analysis tools
- **Algorithm Visualization**: Complete SVL 5.0 compliant visualization system
- **Web Interface**: User-friendly web application for algorithm exploration

## Research Contributions
- Novel algorithm generation approaches using LLMs
- Chain-of-thought reasoning for algorithmic tasks
- Soft matching evaluation metrics
- Comprehensive empirical evaluation on multiple datasets
- SVL 5.0 specification implementation

## File Statistics
- **Total Python Files**: 50
- **Research Code**: 15 files across 5 categories
- **Application Code**: 35 files including algorithm trackers and web components
- **Datasets**: Complete raw and processed datasets
- **Documentation**: Comprehensive README and dataset descriptions

## Citation
If you use this code or datasets in your research, please cite our paper.

## License
[Add your license information here] 