import os
import json
import numpy as np
import faiss
import pickle
import streamlit as st
from rank_bm25 import BM25Okapi
from botocore.exceptions import ClientError

import shared_config

# Constants for retrieval tuning
SEARCH_CANDIDATES = 50
FINAL_CONTEXT_SIZE = 15
CONTEXT_WINDOW_SIZE = 1

class RAGRetriever:
    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client
        self.vector_index = None
        self.docstore = {}
        self.bm25 = None
        self.corpus_ids = []
        
        self._load_indexes()

    def _load_indexes(self):
        faiss_path = os.path.join(shared_config.FAISS_INDEX_PATH, "index.faiss")
        pkl_path = os.path.join(shared_config.FAISS_INDEX_PATH, "index.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            self.vector_index = faiss.read_index(faiss_path)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.docstore = data["docstore"]
            
            self.corpus_ids = sorted(list(self.docstore.keys()))
            tokenized_corpus = [self.docstore[doc_id]["text"].lower().split() for doc_id in self.corpus_ids]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("Indici FAISS e BM25 caricati in memoria.")

    def is_ready(self):
        return self.vector_index is not None and self.bm25 is not None

    def get_embedding(self, text):
        if not text: return [0.0] * 1024
        body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
        try:
            response = self.bedrock_client.invoke_model(
                modelId=shared_config.BEDROCK_EMBEDDING_MODEL, 
                contentType="application/json", 
                accept="application/json", 
                body=body
            )
            emb = json.loads(response['body'].read()).get("embedding")
            return np.array(emb, dtype="float32")
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros(1024, dtype="float32")

    def expand_query(self, query):
        system_prompt = (
            "You are an expert Query Rewriting System optimized for a Hybrid Search architecture (Vector + Keyword) "
            "in the domain of Italian Energy Markets (GME, ARERA, Terna, EU Regulations).\n"
            "Your goal is to rewrite the user's raw question into a polished, semantically rich query that maximizes retrieval performance.\n"
            "1. The output must be a coherent, grammatically correct sentence in Italian. Do NOT output a list of keywords.\n"
            "2. Replace generic terms with specific industry terminology.\n"
            "3. If a concept has a standard acronym, include both the full name and the acronym within the sentence naturally.\n"
            "4. Answer ONLY with the new query, nothing else."
        )
        try:
            haiku_id = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5", "anthropic.claude-3-haiku-20240307-v1:0")
            response = self.bedrock_client.converse(
                modelId=haiku_id,
                messages=[{"role": "user", "content": [{"text": f"Question: {query}"}]}],
                system=[{"text": system_prompt}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 100}
            )
            return response['output']['message']['content'][0]['text'].strip()
        except Exception:
            return query

    def get_expanded_context(self, doc_id, window=1):
        center_doc = self.docstore[doc_id]
        source = center_doc["source"]
        
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
        fusion_scores = {}
        for rank_list in rank_lists:
            for rank, (doc_id, _) in enumerate(rank_list):
                if doc_id not in fusion_scores: 
                    fusion_scores[doc_id] = 0
                fusion_scores[doc_id] += 1 / (k + rank + 1)
        return sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query, search_k=SEARCH_CANDIDATES, final_k=FINAL_CONTEXT_SIZE):
        if not query or not self.is_ready(): 
            return [], query

        search_query = self.expand_query(query)
        
        q_vec = np.array([self.get_embedding(search_query)])
        D, I = self.vector_index.search(q_vec, k=search_k)
        dense_results = [(self.corpus_ids[idx], 0) for idx in I[0] if idx != -1]

        bm25_scores = self.bm25.get_scores(search_query.lower().split())
        bm25_top_idxs = np.argsort(bm25_scores)[::-1][:search_k]
        sparse_results = [(self.corpus_ids[idx], 0) for idx in bm25_top_idxs]

        fused_results = self.reciprocal_rank_fusion([dense_results, sparse_results])
        
        final_context = []
        seen_content = set()
        
        for doc_id, score in fused_results[:final_k]:
            expanded_text = self.get_expanded_context(doc_id, window=CONTEXT_WINDOW_SIZE)
            
            snippet = expanded_text[:100]
            if snippet not in seen_content:
                final_context.append(expanded_text)
                seen_content.add(snippet)
                
        return final_context, search_query

@st.cache_resource
def get_retriever(_bedrock_client):
    return RAGRetriever(_bedrock_client)