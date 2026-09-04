import os
import sys
from huggingface_hub import HfApi

TOKEN = sys.argv[1]
api = HfApi(token=TOKEN)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# afriteva already done as the pipeline test -- skip re-upload.
FULL_FT_MODELS = [
    "mt5", "afrimt5", "nllb", "mbart50",
    "afriteva_v2_large", "mt5_large", "cheetah", "m2m100",
    "m2m100_1.2b", "seamless",
]

LORA_MODELS = [
    "afriteva", "mt5", "afrimt5", "nllb", "mbart50",
    "afriteva_v2_large", "mt5_large", "cheetah", "m2m100",
    "m2m100_1.2b", "seamless", "madlad3b", "t5v11xl", "toucan",
]


def backup(local_dir, repo_id, label):
    if not os.path.isdir(local_dir):
        print(f"SKIP {label}: {local_dir} not found", flush=True)
        return
    print(f"--- START {label}: {local_dir} -> {repo_id} ---", flush=True)
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        ignore_patterns=["checkpoint-*/**", "checkpoint-*"],
    )
    print(f"DONE {label}", flush=True)


for model in FULL_FT_MODELS:
    backup(
        os.path.join(DATA_DIR, f"output_{model}"),
        f"coderGit/eng-pidginedu-backup-{model.replace('.', '-')}",
        f"full-FT {model}",
    )

for model in LORA_MODELS:
    backup(
        os.path.join(DATA_DIR, f"output_lora_{model}"),
        f"coderGit/eng-pidginedu-backup-lora-{model.replace('.', '-')}",
        f"LoRA {model}",
    )

print("ALL BACKUPS COMPLETE", flush=True)
