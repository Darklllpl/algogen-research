#!/bin/bash

# AlgoGen Experiment Reproduction Script
# This script reproduces all major experiments in the paper

echo "Starting AlgoGen experiment reproduction..."

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"

# 1. Data Processing
echo "Step 1: Data Processing"
cd code/data_processing

echo "Creating Gemma datasets..."
python create_gemma_dataset.py

echo "Creating T5 datasets..."
python create_t5_dataset_snapshot.py
python create_t5_native_dataset.py

cd ../..

# 2. Model Training
echo "Step 2: Model Training"
cd code/model_training

echo "Training Gemma models..."
python train_gemma.py

echo "Training delta predictor..."
python train_delta_predictor.py

cd ../..

# 3. Inference
echo "Step 3: Running Inference"
cd code/inference

echo "Running autoregressive chain-of-thought inference..."
python run_cot_autoregressive.py

echo "Running latest autoregressive inference..."
python run_cot_autoregressive_newest.py

echo "Running step-by-step generation..."
python step_by_step_generator.py
python step_by_step_generator_gemma.py

cd ../..

# 4. Evaluation
echo "Step 4: Evaluation"
cd code/evaluation

echo "Running main evaluation..."
python evaluation.py

echo "Running improved evaluation..."
python evaluation_better.py

echo "Running JSONL evaluation..."
python evaluate_jsonl.py
python evaluate_jsonl_gemma.py

echo "Calculating confidence intervals..."
python em_confidence_interval.py

echo "Calculating soft FFI..."
python calculate_soft_ffi.py

cd ../..

# 5. Visualization
echo "Step 5: Visualization"
cd code/visualization

echo "Running renderer..."
python renderer.py

cd ../..

echo "All experiments completed!"
echo "Results are available in datasets/evaluation_results/" 