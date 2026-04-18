from huggingface_hub import HfApi

api = HfApi()

repo_id = "deepak002p/FakeNews-DistilBERT"
folder_path = "distilbert_model/"

print(f"Creating repository {repo_id}...")
try:
    api.create_repo(repo_id=repo_id, exist_ok=True)
except Exception as e:
    print(f"Repo might already exist or error: {e}")

print(f"Uploading folder {folder_path} to {repo_id}...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model"
)

print("Upload complete!")
