# 🇮🇩🌐🤖 IndoWebGen 

## Data Generation
The data generation process utilizes the Alpaca Self-Instruct pipeline, but with the OpenAI chat model (e.g., `gpt-3.5-turbo`). Please note tthat his pipeline does not employ a batch system since the chat model does not support prompt batching.
```bash
python -m generate_data generate_instruction_following_data \
  --output_dir="./" \
  --num_instructions_to_generate=100 \
  --num_instructions_to_generate_per_request=2 \
  --model_name=gpt-3.5-turbo-16k \
  --similarity_threshold=0.6
```

## Fine-tuning
The fine-tuning process utilizes the WizardLM-13B-V1.2 model.

```bash
python finetune.py \
  --hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --base_model_id=codellama/CodeLlama-7b-hf \
  --dataset_id=alimtegar/indowebgen-dataset \
  --output_dir="./indowebgen-7b-4k-lora" \
  --output_model_id=alimtegar/indowebgen-7b-4k-lora \
  --commit_message="Finetune for 3 epochs" \
  --vastai_api_key=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --vastai_instance_id=1234567
  --stop_vastai_instance=0
```

<!-- 1. Clone the Llama-X repository:
```
git clone https://github.com/AetherCortex/Llama-X.git
```
2. Go to to the **Llama-X** directory and install the required libraries:
```
pip install -r requirements.txt
pip install deepspeed==0.9.2 transformers==4.29.2 datasets
```
3. Go to to the **Llama-X/src** directory and download **train_wizardcoder.py** from the WizardLM/WizardCoder repository:
```
wget https://raw.githubusercontent.com/nlpxucan/WizardLM/main/WizardCoder/src/train_wizardcoder.py
```
4. Before training, log in to Hugging Face:
```
huggingface-cli login
```
5. In the **Llama-X/src** directory, execute the following training command:
```
deepspeed train_wizardcoder.py \
    --model_name_or_path "bigcode/starcoder" \
    --data_path "alimtegar/webgen-dataset" \
    --output_dir "./output" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
``` -->
