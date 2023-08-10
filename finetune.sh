#!/bin/bash

# Set up LoRA
git clone https://github.com/tloen/alpaca-lora.git
cd alpaca-lora
awk '{gsub("data_point\\[\"input\"\\]", "None")}1' finetune.py > finetune_.py && mv finetune_.py finetune.py
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
pip install scipy

# Set up vast.ai
if [ ! -z "$2" ]; then
    pip install --upgrade vastai
    vastai set api-key "$2"
else
    echo "Warning: vast.api API key not provided. Skipping vast.ai setup."
fi

# Finetune model
huggingface-cli login --token "$1"
python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path 'alimtegar/webgen-dataset-2' \
    --output_dir './lora-alpaca' \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 8 \
    --learning_rate 1e-4 \
    --cutoff_len 2048 \
    --val_set_size 20 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length

# Push model to Hugging Face
python push_to_hf.py

# Stop vast.ai instance
if [ ! -z "$2" ] && [ ! -z "$3" ]; then
    vastai stop instance "$3"
else
    echo "Warning: vast.api API key and instance ID not provided. Skipping vast.ai instance stopping."
fi
