import os
import json
import boto3
import numpy as np
import faiss
import pickle
import concurrent.futures
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from botocore.config import Config
import time
import random

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Files
INPUT_FILE = os.path.join(CURRENT_DIR, "test_rag.jsonl")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "results/gemma-27b.json")

FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index_rag", "index.faiss")
PKL_INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index_rag", "index.pkl")

# Models
INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_GEMMA_3_27B") 
HELPER_INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5") 
EMBEDDING_MODEL_ID = os.getenv("EMBEDDINGS_MODEL_TITAN_TEXT_V2")
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

# --- HYPERPARAMETERS FOR THESIS ---
# We retrieve more chunks (Top-K 15) but strictly filter them
NUM_ANSWERS = 5
SEARCH_CANDIDATES = 50
FINAL_CONTEXT_SIZE = 15      
CONTEXT_WINDOW_SIZE = 1

INFERENCE_CONFIG = {
    "maxTokens": 1024,
    "temperature": 0.5, 
    "topP": 0.9
}

# Bedrock Client with Adaptive Retry
retry_config = Config(
    region_name=REGION_NAME,
    retries={'max_attempts': 8, 'mode': 'adaptive'},
    read_timeout=900,
    connect_timeout=10
)
bedrock_client_global = boto3.client(service_name='bedrock-runtime', config=retry_config)

# ==============================================================================
# 1. ADVANCED RETRIEVER (HYBRID + RRF)
# ==============================================================================
class RAGRetriever:
    def __init__(self):
        print("Loading FAISS & Metadata...")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"Index not found at {FAISS_INDEX_PATH}. Run Step 2.")
            
        self.vector_index = faiss.read_index(FAISS_INDEX_PATH)
        
        with open(PKL_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
            self.docstore = data["docstore"]
            self.source_map = data["source_map"]
            
        print("Building BM25 Index (Sparse)...")
        self.corpus_ids = sorted(list(self.docstore.keys()))
        # Tokenize once for BM25
        tokenized_corpus = [
            self.docstore[doc_id]["text"].lower().split() 
            for doc_id in self.corpus_ids
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_embedding(self, text):
        """Generates normalized embedding."""
        if not text: return [0.0] * 1024
        body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
        try:
            response = bedrock_client_global.invoke_model(
                modelId=EMBEDDING_MODEL_ID, contentType="application/json", accept="application/json", body=body
            )
            emb = json.loads(response['body'].read()).get("embedding")
            return np.array(emb, dtype="float32")
        except Exception as e:
            print(f"Embedding Error: {e}")
            return np.zeros(1024, dtype="float32")

    def expand_query(self, query):
        """
        Riscrive la query utente aggiungendo termini tecnici/legali.
        Migliora drasticamente la Recall.
        """
        system_prompt = (
            "You are an expert AI for Legal Search Query Optimization.\n"
            "Rewrite the user's question into a search query optimized for a Vector Database "
            "containing Italian energy trade laws (REMIT, MIFID, etc.).\n"
            "Rules:\n"
            "1. Remove conversational filler.\n"
            "2. Add specific Italian legal synonyms (e.g. 'fine' -> 'sanzione amministrativa', 'cheating' -> 'manipolazione').\n"
            "3. Output ONLY the rewritten query text in Italian."
        )
        
        try:
            # Usiamo un modello veloce per l'espansione (o lo stesso dell'inferenza)
            response = bedrock_client_global.converse(
                modelId=HELPER_INFERENCE_PROFILE,
                messages=[{"role": "user", "content": [{"text": f"Question: {query}"}]}],
                system=[{"text": system_prompt}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 100}
            )
            rewritten = response['output']['message']['content'][0]['text'].strip()
            # Safety check: se il modello impazzisce e scrive troppo, teniamo l'originale
            if len(rewritten) > len(query) * 4: return query 
            return rewritten
        except:
            return query
    
    def get_expanded_context(self, doc_id, window=1):
        """
        Retrieves the chunk at doc_id PLUS its neighbors to restore flow.
        """
        center_doc = self.docstore[doc_id]
        source = center_doc["source"]
        
        # Collect IDs: [prev_window ... center ... next_window]
        ids_to_fetch = []
        for i in range(window, 0, -1):
            ids_to_fetch.append(doc_id - i)
        ids_to_fetch.append(doc_id)
        for i in range(1, window + 1):
            ids_to_fetch.append(doc_id + i)
            
        chunks = []
        for pid in ids_to_fetch:
            if pid in self.docstore and self.docstore[pid]["source"] == source:
                chunks.append(self.docstore[pid]["text"])
        
        return "\n".join(chunks)

    def reciprocal_rank_fusion(self, rank_lists, k=60):
        """
        Standard RRF algorithm. 
        Combines multiple ranked lists into a single score.
        score = sum(1 / (k + rank))
        """
        fusion_scores = {}
        for rank_list in rank_lists:
            for rank, (doc_id, _) in enumerate(rank_list):
                if doc_id not in fusion_scores:
                    fusion_scores[doc_id] = 0
                fusion_scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by RRF score descending
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def search(self, query, search_k=SEARCH_CANDIDATES, final_k=FINAL_CONTEXT_SIZE):
        """
        Performs Hybrid Search using Reciprocal Rank Fusion.
        """
        if not query: return [], query

        # 1. Query Expansion
        search_query = self.expand_query(query)
        
        # 2. Dense Search (Vector) - Peschiamo AMPIO (es. 60 candidati)
        q_vec = np.array([self.get_embedding(search_query)])
        D, I = self.vector_index.search(q_vec, k=search_k)
        dense_results = [(self.corpus_ids[idx], 0) for idx in I[0] if idx != -1]

        # 3. Sparse Search (BM25) - Peschiamo AMPIO (es. 60 candidati)
        bm25_scores = self.bm25.get_scores(search_query.lower().split())
        bm25_top_idxs = np.argsort(bm25_scores)[::-1][:search_k]
        sparse_results = [(self.corpus_ids[idx], 0) for idx in bm25_top_idxs]

        # 4. Fusion (RRF) - Classifica i 60+60 candidati
        fused_results = self.reciprocal_rank_fusion([dense_results, sparse_results])
        
        # 5. Selection & Context Retrieval - Prendiamo solo i TOP per l'LLM
        final_context = []
        seen_content = set()
        
        for doc_id, score in fused_results[:final_k]: # <--- QUI AVVIENE IL TAGLIO INTELLIGENTE
            expanded_text = self.get_expanded_context(doc_id, window=CONTEXT_WINDOW_SIZE)
            
            # Deduplicazione base (primi 100 char)
            snippet = expanded_text[:100]
            if snippet not in seen_content:
                final_context.append(expanded_text)
                seen_content.add(snippet)
                
        return final_context, search_query

# ==============================================================================
# 2. GENERATION LOGIC
# ==============================================================================

def generate_rag_response(query, context_list, original_sys_prompt, thread_id=0):
    if not query: return "Error: No Query"

    # 1. Format Context into XML for Llama/Claude (Best Practice)
    formatted_context = ""
    for i, ctx in enumerate(context_list):
        formatted_context += f"<doc id='{i+1}'>{ctx}</doc>\n"

    # 2. Construct System Prompt
    # We Inject RAG instructions but KEEP the Persona
    base_system = original_sys_prompt
    
    rag_system_prompt = (
        f"{base_system}\n\n"
        "### RAG INSTRUCTIONS (STRICT COMPLIANCE REQUIRED)\n"
        "You have access to a set of retrieved documents enclosed in <context> tags.\n"
        "1. Answer the user's question using ONLY the information provided in the <context>. DON'T USE YOUR KNOWLEDGE.\n"
        "2. Combine information from multiple chunks to answer fully.\n"
        "3. If the exact wording isn't present but the answer can be logically inferred from the context (e.g. synonyms), do so.\n"
        "4. If the answer is not present in the <context>, you must reply exactly: [RISPOSTA_NON_TROVATA]\n"
        "5. Cite the documents where possible (eg 'Secondo il documento...').\n"
    )

    # 3. Construct User Message
    user_message = (
        f"<context>\n{formatted_context}\n</context>\n\n"
        f"Domanda: {query}"
    )

    # 4. Inference (Converse API)
    # We try up to 3 times for robustness
    for attempt in range(3):
        try:
            response = bedrock_client_global.converse(
                modelId=INFERENCE_PROFILE,
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                system=[{"text": rag_system_prompt}],
                inferenceConfig=INFERENCE_CONFIG
            )
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            time.sleep(2)
            if attempt == 2:
                return f"[ERROR] Generation failed: {str(e)}"

def process_single_entry(entry, retriever):
    # Extract data from Q&A structure
    sys_prompt = ""
    user_query = ""
    gold_answer = ""
    
    for msg in entry.get("messages", []):
        if msg['role'] == "system": sys_prompt = msg['content']
        elif msg['role'] == "user": user_query = msg['content']
        elif msg['role'] == "assistant": gold_answer = msg['content']

    if not user_query: return None

    # A. Retrieve
    retrieved_docs, actual_search_query = retriever.search(user_query)

    # B. Generate (Parallel for robustness check)
    responses = []
    # We generate 5 answers to check stability (as per your request)
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ANSWERS) as executor:
        futures = [
            executor.submit(generate_rag_response, user_query, retrieved_docs, sys_prompt, i) 
            for i in range(NUM_ANSWERS)
        ]
        for f in concurrent.futures.as_completed(futures):
            responses.append(f.result())

    return {
        "id": entry.get("index"),
        "original_question": user_query,
        "new_query": actual_search_query,
        "gold_answer": gold_answer,
        "score": entry.get("score"),
        "is_grounded": entry.get("is_grounded"),
        "retrieved_context": retrieved_docs,
        "generated_responses": responses,
        "model_profile": INFERENCE_PROFILE
    }

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================
def main():
    print(f"--- THESIS RAG GENERATION ---")
    print(f"Model: {INFERENCE_PROFILE}")
    print(f"Retrieval: Hybrid RRF (Top-{FINAL_CONTEXT_SIZE})")
    
    # Init Retriever
    retriever = RAGRetriever()

    # Load Data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Handle both list of objects and line-delimited JSON
        try:
            data = json.load(f)
        except:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # Run Inference
    results = []
    # Using tqdm for progress tracking
    for entry in tqdm(data, desc="Generating Answers"):
        res = process_single_entry(entry, retriever)
        if res:
            results.append(res)
            
            # Streaming save (safety first)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, ensure_ascii=False, indent=4)

    print(f"Complete. Saved {len(results)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()