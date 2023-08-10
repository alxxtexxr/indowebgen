from huggingface_hub import HfApi


api = HfApi()
api.upload_folder(
    folder_path='./lora-alpaca/',
    repo_id='alimtegar/webgen-2-lora',
    allow_patterns='*',
    delete_patterns='*',
)