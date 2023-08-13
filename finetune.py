import os
import fire
from huggingface_hub import HfApi


def main(
  hf_token, 
  base_model_id, 
  dataset_id, 
  output_dir, 
  output_model_id,
  commit_message,
  batch_size=32,
  num_epochs=3,
  cutoff_len=2048, 
  lora_r=8,
  lora_target_modules='[q_proj,v_proj]',
  vastai_api_key=None,
  vastai_instance_id=None,
):
  # Set up LoRA
  os.system('git clone https://github.com/tloen/alpaca-lora.git')
  os.chdir('alpaca-lora')
  os.system('''awk \'{gsub("data_point\\\\[\\\"input\\\"\\\\]", "None")}1\' \\
    finetune.py > finetune_custom.py''')
  
  os.system('pip install -r requirements.txt')
  os.system('pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08')
  os.system('pip install scipy')

  # Set up vast.ai
  if vastai_api_key:
    os.system(f'vastai set api-key {vastai_api_key}')

  # Finetune model
  os.system(f'huggingface-cli login --token {hf_token}')
  os.system(f'''python finetune_custom.py \\
    --base_model "{base_model_id}" \\
    --data_path "{dataset_id}" \\
    --output_dir "{output_dir}" \\
    --batch_size {batch_size} \\
    --micro_batch_size 4 \\
    --num_epochs {num_epochs} \\
    --learning_rate 1e-4 \\
    --cutoff_len {cutoff_len} \\
    --val_set_size 20 \\
    --lora_r {lora_r} \\
    --lora_alpha 16 \\
    --lora_dropout 0.05 \\
    --lora_target_modules "{lora_target_modules}" \\
    --train_on_inputs \\
    --group_by_length
  ''')

  # Push model to Hugging Face
  api = HfApi()
  api.upload_folder(
    folder_path=output_dir,
    repo_id=output_model_id,
    allow_patterns='*',
    delete_patterns='*',
    commit_message=commit_message
  )

  # Stop vast.ai instance
  if vastai_api_key and vastai_instance_id:
	  os.system(f'vastai stop instance {vastai_instance_id}')
  

if __name__ == '__main__':
  fire.Fire(main)