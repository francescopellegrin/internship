import os
import sys
from huggingface_hub import snapshot_download

# 0. Set HF_HOME explicitly to your WORK cache
work_dir = "/leonardo_work/AIFAC_F01_260/"
if not work_dir:
    print("Error: WORK variable is not set.")
    sys.exit(1)

os.environ["HF_HOME"] = os.path.join(work_dir, ".cache", "huggingface")

# 1. Setup paths
# Verifica l'ID su HF
model_id = "google/gemma-3-27b-it"

# New destination: $WORK/offline_model/gemma-3-27b
local_folder = os.path.join(work_dir, "offline_models", "gemma-3-27b")

print(f"--- STARTING DOWNLOAD ---")
print(f"HF Cache:    {os.environ['HF_HOME']}")
print(f"Model:       {model_id}")
print(f"Destination: {local_folder}")
print("This download requires explicit Hugging Face access (Gated Repo).")

# 2. Download directly to the target folder
try:
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_folder,
        local_dir_use_symlinks=False,  # Force real files
        resume_download=True,          
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] 
    )
    
    print(f"--- SUCCESS ---")
    print(f"Model ready at: {local_folder}")

except Exception as e:
    print(f"--- ERROR ---")
    print(f"{e}")
    print("Tip: If this is an authentication error, run 'huggingface-cli login' inside your ft_env.")