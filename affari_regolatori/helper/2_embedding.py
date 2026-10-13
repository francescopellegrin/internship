import os
import json
import boto3
import numpy as np
import faiss
import pickle
import time
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# --- Configuration & Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

# FIX: Corretto il nome del file di input per coincidere con lo step 1
INPUT_FILE = os.path.join(CURRENT_DIR, "chunks_smart.json")

OUTPUT_FOLDER = os.path.join(CURRENT_DIR, "faiss_index")
INDEX_FILE = os.path.join(OUTPUT_FOLDER, "index.faiss")
PKL_FILE = os.path.join(OUTPUT_FOLDER, "index.pkl")

# Model Settings
EMBEDDING_MODEL_ID = os.getenv("EMBEDDINGS_MODEL_TITAN_TEXT_V2")
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

# Bedrock Client
session = boto3.Session(region_name=REGION_NAME)
client = session.client("bedrock-runtime")

def normalize_vector(v):
    """
    Normalizes a vector to unit length (L2 norm = 1).
    Essential for Dot Product to equal Cosine Similarity.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_embedding_safe(text, retries=3):
    """
    Invokes Amazon Titan Text Embeddings V2 with retry logic.
    """
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024, 
        "normalize": True 
    })

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            response_body = json.loads(response['body'].read())
            embedding = response_body.get("embedding")
            
            # Double safety: Normalize locally ensuring consistency for FAISS IP
            return normalize_vector(np.array(embedding, dtype='float32'))

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
            else:
                print(f"AWS Error: {e}")
                return None
        except Exception as e:
            print(f"General Error: {e}")
            return None
    
    return None

def create_faiss_index():
    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")

    # 2. Load Processed Chunks
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run Step 1 first.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks.")

    # 3. Generate Embeddings & Build Maps
    embeddings = []
    docstore = {} 
    source_map = defaultdict(list) 

    print("Generating embeddings with Amazon Titan V2...")
    
    valid_count = 0
    # Use tqdm to show progress bar
    for chunk in tqdm(chunks, desc="Embedding"):
        text = chunk.get("text", "") # Questo è il testo con HEADER (Document: ...)
        source = chunk.get("source", "unknown")
        
        if not text:
            continue
            
        vector = get_embedding_safe(text)
        
        if vector is not None:
            embeddings.append(vector)
            
            # Save metadata mapping
            docstore[valid_count] = {
                "text": text,
                "source": source,
                "chunk_index": chunk.get("chunk_index"),
                "metadata": chunk.get("metadata", {})
            }
            
            # Reverse index for filtering
            source_map[source].append(valid_count)
            valid_count += 1

    if not embeddings:
        print("No embeddings generated. Exiting.")
        return

    # 4. Create & Save FAISS Index using INNER PRODUCT (Cosine Similarity)
    embedding_matrix = np.array(embeddings).astype('float32')
    dimension = embedding_matrix.shape[1]
    
    print(f"Creating FAISS Index (Dimension: {dimension}, Metric: Inner Product)...")
    
    # IndexFlatIP = Inner Product. 
    # Since vectors are normalized, this is exact Cosine Similarity.
    index = faiss.IndexFlatIP(dimension) 
    index.add(embedding_matrix)
    
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved FAISS index ({index.ntotal} vectors) to {INDEX_FILE}")

    # 5. Save Metadata & Source Map
    full_store = {
        "docstore": docstore,      
        "source_map": dict(source_map)   
    }
    
    with open(PKL_FILE, "wb") as f:
        pickle.dump(full_store, f)
        
    print(f"Saved optimized metadata store to {PKL_FILE}")
    print(f"  - Unique sources indexed: {len(source_map.keys())}")

if __name__ == "__main__":
    create_faiss_index()