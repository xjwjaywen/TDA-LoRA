#!/bin/bash
# Run all experiments for TDA-LoRA paper
# Usage: bash scripts/run_all.sh [GPU_ID] [MODEL_PATH]

GPU=${1:-0}
MODEL=${2:-"./models/AI-ModelScope/stable-diffusion-2-1-base"}

export CUDA_VISIBLE_DEVICES=$GPU

METHODS=(
    "lora_only"
    "timestep_only"
    "domain_only"
    "layer_only"
    "no_layer"
    "tda_lora"
)

for class in "001.Black_footed_Albatross" "002.Laysan_Albatross"; do
    for shots in 10 5; do
        for method in "${METHODS[@]}"; do
            echo ">>> $method | $class | ${shots}-shot"
            python train.py \
                --dataset cub200 \
                --class_name "$class" \
                --num_shots "$shots" \
                --method "$method" \
                --model_path "$MODEL"
            echo "<<< Done: $method"
        done
    done
done

echo "All experiments complete!"
