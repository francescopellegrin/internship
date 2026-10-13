import os
from dotenv import load_dotenv

# Get the absolute path to the directory where this script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Go up one level

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

REGION = os.getenv("REGION", "eu-west-1")

BEDROCK_EMBEDDING_MODEL = os.getenv("EMBEDDINGS_MODEL_TITAN_TEXT_V2")
INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_QWEN3_NEXT_80B")
MODEL_NAME = "qwen3-next-80b"
FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "helper/faiss_index")
INDEX_METADATA_PATH = os.path.join(FAISS_INDEX_PATH, "index_metadata.json")
RERANKER_TOP_K = 5

# Join that script directory path with the chat_history folder name
CHAT_HISTORY_DIR = os.path.join(CURRENT_DIR, "chat_history")

# Create chat history directory if it doesn't exist
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)