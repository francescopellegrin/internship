import os
import tempfile
import shutil
import json
import numpy as np
import faiss
import pickle
import time
import concurrent.futures
import fitz  # PyMuPDF
import streamlit as st
import boto3
from botocore.exceptions import ClientError

import shared_config

# --- Constants ---
# Using the models defined in your environment
CHUNKING_MODEL_ID = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5", "anthropic.claude-3-haiku-20240307-v1:0")
LLM_BUFFER_SIZE = 12000
MAX_WORKERS = 5

def clean_page_content_vision(page_bytes, bedrock_client):
    system_prompt = (
        "You are a specialized data cleaning assistant for legal documents.\n"
        "Transcribe the provided PDF page text accurately.\n"
        "RULES:\n"
        "1. Remove headers, footers, page numbers, and logos.\n"
        "2. Merge hyphenated words (e.g., 're- gulation' -> 'regulation').\n"
        "3. Output ONLY the clean text. No conversational filler."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"document": {"name": "page", "format": "pdf", "source": {"bytes": page_bytes}}},
                {"text": "Transcribe this page."}
            ]
        }
    ]
    try:
        response = bedrock_client.converse(
            modelId=CHUNKING_MODEL_ID,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0}
        )
        return response['output']['message']['content'][0]['text'].strip()
    except Exception:
        return None

def llm_semantic_split(text_buffer, source_doc, current_chunk_index, bedrock_client, retry_count=0):
    system_prompt = (
        "You are a Semantic Chunker for RAG systems. Split the text into logical, self-contained chunks.\n"
        "RULES:\n"
        "1. JSON Output ONLY. No intro/outro text.\n"
        "2. Structure: { 'chunks': [ { 'title': '...', 'content': '...' } ], 'remainder': '...' }\n"
        "3. ESCAPE ALL QUOTES inside content strings carefully.\n"
        "4. If text ends abruptly, put that part in 'remainder'.\n"
        "5. Avoid creating chunks smaller than 200 characters unless it's a standalone definition."
    )
    safe_text = text_buffer.replace('"', "'")
    user_msg = f"Input Text ({len(text_buffer)} chars):\n<text>\n{safe_text}\n</text>\n\nGenerate JSON."

    try:
        response = bedrock_client.converse(
            modelId=CHUNKING_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 4096}
        )
        
        output_text = response['output']['message']['content'][0]['text']
        
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON brackets found")
            
        json_str = output_text[start:end]
        data = json.loads(json_str)
        
        formatted_chunks = []
        for item in data.get("chunks", []):
            content_str = item.get('content', '').strip()
            if len(content_str) < 20: continue

            content_with_header = (
                f"DOCUMENT: {source_doc}\n"
                f"SECTION: {item.get('title', 'Segment')}\n"
                f"ID: {current_chunk_index}\n"
                f"CONTENT:\n{content_str}"
            )
            
            formatted_chunks.append({
                "source": source_doc,
                "chunk_index": current_chunk_index,
                "text": content_with_header,
                "metadata": {
                    "source": source_doc,
                    "title": item.get('title'),
                    "original_text": content_str
                }
            })
            current_chunk_index += 1
            
        return formatted_chunks, data.get("remainder", "")

    except json.JSONDecodeError:
        if retry_count < 2:
            return llm_semantic_split(text_buffer, source_doc, current_chunk_index, bedrock_client, retry_count + 1)
        else:
            return [], text_buffer
    except Exception as e:
        return [], text_buffer

def process_document_pipeline(filepath, filename, bedrock_client, status_container):
    doc = fitz.open(filepath)
    
    tasks = []
    for i in range(len(doc)):
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        tasks.append((new_doc.tobytes(), i))
        new_doc.close()
    doc.close()

    cleaned_pages = {}
    status_container.write(f"Avvio Vision OCR su {len(tasks)} pagine per {filename}...")
    
    progress_bar = st.progress(0)
    completed_pages = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(clean_page_content_vision, t[0], bedrock_client): i for i, t in enumerate(tasks)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            text = future.result()
            if text:
                cleaned_pages[i] = text
            completed_pages += 1
            progress_bar.progress(completed_pages / len(tasks))
            
    progress_bar.empty()

    full_text_stream = ""
    for i in range(len(tasks)):
        if i in cleaned_pages:
            full_text_stream += cleaned_pages[i] + "\n\n"

    status_container.write(f"Chunking semantico in corso su {len(full_text_stream)} caratteri...")
    
    final_chunks = []
    chunk_idx_counter = 0
    buffer = ""
    words = full_text_stream.split()
    current_word_idx = 0
    
    chunk_progress = st.progress(0)
    
    while current_word_idx < len(words):
        while len(buffer) < LLM_BUFFER_SIZE and current_word_idx < len(words):
            buffer += words[current_word_idx] + " "
            current_word_idx += 1
        
        is_last_batch = (current_word_idx >= len(words))
        
        new_chunks, remainder = llm_semantic_split(buffer, filename, chunk_idx_counter, bedrock_client)
        
        if new_chunks:
            final_chunks.extend(new_chunks)
            chunk_idx_counter += len(new_chunks)
            
        chunk_progress.progress(current_word_idx / len(words))
        
        if is_last_batch:
            if remainder.strip():
                final_chunks.append({
                    "source": filename,
                    "chunk_index": chunk_idx_counter,
                    "text": f"DOCUMENT: {filename}\nSECTION: Final Segment\nCONTENT:\n{remainder}",
                    "metadata": {"source": filename, "original_text": remainder}
                })
            break
        else:
            buffer = remainder + " " 
            if len(remainder) > len(buffer) * 0.9:
                cut_point = len(buffer) // 2
                processed_part = buffer[:cut_point]
                buffer = buffer[cut_point:]
                final_chunks.append({
                    "source": filename,
                    "chunk_index": chunk_idx_counter,
                    "text": f"DOCUMENT: {filename}\nSECTION: Segment {chunk_idx_counter}\nCONTENT:\n{processed_part}",
                    "metadata": {"source": filename, "original_text": processed_part}
                })
                chunk_idx_counter += 1

    chunk_progress.empty()
    status_container.write(f"Creati {len(final_chunks)} semantic chunks per {filename}.")
    return final_chunks

def read_and_split_uploaded_pdfs(uploaded_files, bedrock_client):
    all_chunks = []
    if not uploaded_files:
        st.warning("Nessun file caricato.")
        return []
    
    with st.status("Elaborazione documenti legali (Smart Chunking)...", expanded=True) as status:
        temp_dir = tempfile.mkdtemp()
        try:
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                status.write(f"Inizio elaborazione file: {uploaded_file.name}")
                doc_chunks = process_document_pipeline(temp_file_path, uploaded_file.name, bedrock_client, status)
                all_chunks.extend(doc_chunks)
                
            status.update(label=f"Chunking completato. Totale frammenti: {len(all_chunks)}", state="complete", expanded=False)
            return all_chunks
        except Exception as e:
            st.error(f"Errore durante il parsing del PDF: {e}")
            return []
        finally:
            shutil.rmtree(temp_dir)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_embedding_safe(text, bedrock_client, retries=3):
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024, 
        "normalize": True 
    })
    for attempt in range(retries):
        try:
            response = bedrock_client.invoke_model(
                modelId=shared_config.BEDROCK_EMBEDDING_MODEL,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            response_body = json.loads(response['body'].read())
            embedding = response_body.get("embedding")
            return normalize_vector(np.array(embedding, dtype='float32'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                time.sleep((2 ** attempt) + 1)
            else:
                return None
        except Exception:
            return None
    return None

def create_vector_store(document_chunks, bedrock_client):
    if not document_chunks:
        st.error("Nessun frammento disponibile per l'embedding.")
        return False
        
    embeddings = []
    docstore = {} 
    source_map = {} 

    with st.status("Generazione Embeddings e Indice FAISS...", expanded=True) as status:
        valid_count = 0
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(document_chunks):
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            
            if not text:
                progress_bar.progress((i + 1) / len(document_chunks))
                continue
                
            vector = get_embedding_safe(text, bedrock_client)
            
            if vector is not None:
                embeddings.append(vector)
                
                docstore[valid_count] = {
                    "text": text,
                    "source": source,
                    "chunk_index": chunk.get("chunk_index"),
                    "metadata": chunk.get("metadata", {})
                }
                
                if source not in source_map:
                    source_map[source] = []
                source_map[source].append(valid_count)
                
                valid_count += 1
                
            progress_bar.progress((i + 1) / len(document_chunks))
            
        progress_bar.empty()
        
        if not embeddings:
            st.error("Nessun embedding generato.")
            return False

        try:
            embedding_matrix = np.array(embeddings).astype('float32')
            dimension = embedding_matrix.shape[1]
            
            status.write("Salvataggio indice FAISS (Inner Product)...")
            index = faiss.IndexFlatIP(dimension) 
            index.add(embedding_matrix)
            
            if not os.path.exists(shared_config.FAISS_INDEX_PATH):
                os.makedirs(shared_config.FAISS_INDEX_PATH)
                
            faiss_file = os.path.join(shared_config.FAISS_INDEX_PATH, "index.faiss")
            faiss.write_index(index, faiss_file)
            
            status.write("Salvataggio metadati e dizionario corpus per BM25...")
            pkl_file = os.path.join(shared_config.FAISS_INDEX_PATH, "index.pkl")
            full_store = {
                "docstore": docstore,      
                "source_map": source_map   
            }
            with open(pkl_file, "wb") as f:
                pickle.dump(full_store, f)
                
            status.update(label="Indicizzazione completata con successo.", state="complete", expanded=False)
            return True
        except Exception as e:
            st.error(f"Errore critico durante la creazione dell'indice: {e}")
            return False

def save_index_metadata(file_names):
    if not os.path.exists(shared_config.FAISS_INDEX_PATH):
        os.makedirs(shared_config.FAISS_INDEX_PATH)
            
    metadata = {"source_files": file_names}
    try:
        with open(shared_config.INDEX_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Errore durante il salvataggio dei metadati dell'indice: {e}")
        return False

def load_index_metadata():
    if os.path.exists(shared_config.INDEX_METADATA_PATH):
        try:
            with open(shared_config.INDEX_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            return metadata.get("source_files", [])
        except Exception as e:
            st.error(f"Errore durante il caricamento dei metadati dell'indice: {e}")
            return []
    return []