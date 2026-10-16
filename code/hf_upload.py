from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

login()

# Point to your local parent directory containing mtg_cards/, pokemon_cards/, yugioh_cards/
DATA_DIR = Path("../data")
REPO_ID = "bmf4295/CSCI635_Final_Project"

print("Loading local image folder and converting to dataset...")
dataset = load_dataset("imagefolder", data_dir=str(DATA_DIR))

print(f"Uploading Parquet shards to Hugging Face ({REPO_ID})...")
# max_shard_size ensures Parquet chunks stay well under repo limits
dataset.push_to_hub(REPO_ID, private=True, max_shard_size="500MB")
print("Upload successful!")