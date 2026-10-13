import os
import json
import boto3
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from botocore.config import Config

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

INPUT_FILE = os.path.join(CURRENT_DIR, "test_rag.jsonl") 
OUTPUT_FILE = os.path.join(CURRENT_DIR, "retrieval.json")

FAISS_INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index_rag", "index.faiss")
PKL_INDEX_PATH = os.path.join(CURRENT_DIR, "faiss_index_rag", "index.pkl")

HELPER_MODEL_ID = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDINGS_MODEL_TITAN_TEXT_V2")
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

# --- HYPERPARAMETERS FISSI ---
SEARCH_CANDIDATES = 50
FINAL_CONTEXT_SIZE = 15
CONTEXT_WINDOW_SIZE = 1

# Bedrock Client
retry_config = Config(region_name=REGION_NAME, retries={'max_attempts': 8, 'mode': 'adaptive'})
bedrock_client = boto3.client(service_name='bedrock-runtime', config=retry_config)

class RAGRetriever:
    def __init__(self):
        print("Loading FAISS & Metadata...")
        self.vector_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(PKL_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
            self.docstore = data["docstore"]
        
        print("Building BM25 Index...")
        self.corpus_ids = sorted(list(self.docstore.keys()))
        tokenized_corpus = [self.docstore[doc_id]["text"].lower().split() for doc_id in self.corpus_ids]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_embedding(self, text):
        if not text: return [0.0] * 1024
        body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
        try:
            response = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL_ID, contentType="application/json", accept="application/json", body=body
            )
            emb = json.loads(response['body'].read()).get("embedding")
            return np.array(emb, dtype="float32")
        except:
            return np.zeros(1024, dtype="float32")

    def expand_query(self, query):
        system_prompt = (
            """
            You are an expert Query Rewriting System optimized for a Hybrid Search architecture (Vector + Keyword) in the domain of Italian Energy Markets (GME, ARERA, Terna, EU Regulations).

            Your goal is to rewrite the user's raw question into a polished, semantically rich query that maximizes retrieval performance for:
            1. Amazon Titan v2 (which requires natural language structure and semantic context).
            2. BM25 (which requires exact matching of technical acronyms and specific legal terminology).

            ### Guidelines:
            1. The output must be a coherent, grammatically correct sentence (or question). Do NOT output a list of keywords. This preserves the vector quality.
            2. Replace generic terms with specific industry terminology.
            3. If a concept has a standard acronym, include both the full name and the acronym within the sentence naturally.
            - Example: "Market Coupling" -> "Market Coupling (PCR)".
            4. The output must be in **Italian** to match the target document language.
            5. Answer only with the new query, nothing else.
            """
        )
        try:
            response = bedrock_client.converse(
                modelId=HELPER_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": f"Question: {query}"}]}],
                system=[{"text": system_prompt}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 100}
            )
            return response['output']['message']['content'][0]['text'].strip()
        except:
            return query

    def get_expanded_context(self, doc_id, window=1):
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
                if doc_id not in fusion_scores: fusion_scores[doc_id] = 0
                fusion_scores[doc_id] += 1 / (k + rank + 1)
        return sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)

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


def main():
    print(f"--- PREPARING RETRIEVAL DATASET ---")
    retriever = RAGRetriever()

    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    prepared_data = []
    
    for entry in tqdm(data, desc="Retrieving Contexts"):
        user_query = ""
        gold_answer = ""
        # Estrazione flessibile
        if "messages" in entry:
            for msg in entry["messages"]:
                if msg['role'] == "user": user_query = msg['content']
                elif msg['role'] == "assistant": gold_answer = msg['content']
        else:
            user_query = entry.get("original_question", "")
            gold_answer = entry.get("gold_answer", "")

        if not user_query: continue

        # --- RETRIEVAL ---
        retrieved_docs, new_query = retriever.search(user_query)

        prepared_entry = {
            "id": entry.get("id", entry.get("index")),
            "original_question": user_query,
            "new_query": new_query,
            "gold_answer": gold_answer,
            "score": entry.get("score"),
            "is_grounded": entry.get("is_grounded"),
            "retrieved_context": retrieved_docs
        }
        prepared_data.append(prepared_entry)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()