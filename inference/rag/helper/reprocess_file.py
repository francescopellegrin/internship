import os
import json
import boto3
import re
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

# --- CONFIGURAZIONE ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

SOURCE_DOCS_DIR = os.path.join(CURRENT_DIR, "documents")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "chunks_smart.json") 

# LISTA DEI FILE DA RIPROCESSARE (Quelli che avevano fallito)
# Devono esistere nella cartella SOURCE_DOCS_DIR
FILES_TO_REPROCESS = [
    "Legge del 30 ottobre 2014.pdf",
    "REMIT Regolamento.pdf",
    "REMIT updated.pdf",         
    "ARERA Sanzione 2022_306.pdf",
    "ARERA Sanzione 2022_307.pdf"
]

# Modello
CHUNKING_MODEL_ID = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5") 
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

LLM_BUFFER_SIZE = 12000 
MAX_WORKERS = 5

# --- Bedrock Client ---
session = boto3.Session(region_name=REGION_NAME)
client = session.client("bedrock-runtime")

def clean_page_content_vision(page_bytes):
    """ Step 1: Vision Cleaning """
    system_prompt = (
        "You are a specialized data cleaning assistant for legal documents.\n"
        "Transcribe the provided PDF page text accurately.\n"
        "RULES:\n"
        "1. Remove headers, footers, page numbers, and logos.\n"
        "2. Merge hyphenated words (e.g., 're- gulation' -> 'regulation').\n"
        "3. Output ONLY the clean text."
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
        response = client.converse(
            modelId=CHUNKING_MODEL_ID,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0}
        )
        return response['output']['message']['content'][0]['text'].strip()
    except Exception:
        return None

def llm_semantic_split(text_buffer, source_doc, current_chunk_index, retry_count=0):
    """
    Step 2: Chunking 'Blindato' (NO JSON ERRORI).
    """
    system_prompt = (
        "You are a Semantic Chunker for legal documents.\n"
        "Your task is to split the input text into logical, self-contained chunks (e.g., Articles, Definitions).\n\n"
        "STRICT OUTPUT FORMAT:\n"
        "1. Use '###CHUNK_SEP###' to separate chunks.\n"
        "2. Inside each chunk, structure it exactly like this:\n"
        "   TITLE: <Short Title>\n"
        "   CONTENT: <The text>\n"
        "3. CLEANING: Replace any double quotes (\") inside the text with single quotes (').\n"
        "4. If the text ends abruptly (mid-sentence), place that part at the very end after '###REMAINDER###'."
    )

    safe_text = text_buffer.replace('"', "'")
    user_msg = f"Input Text from {source_doc}:\n<text>\n{safe_text}\n</text>\n\nBegin output:"

    try:
        response = client.converse(
            modelId=CHUNKING_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 4096}
        )
        
        output_text = response['output']['message']['content'][0]['text']
        
        # Parsing Residuo
        remainder = ""
        if "###REMAINDER###" in output_text:
            parts = output_text.split("###REMAINDER###")
            output_text = parts[0]
            if len(parts) > 1:
                remainder = parts[1].strip()

        # Parsing Chunk
        raw_chunks = output_text.split("###CHUNK_SEP###")
        formatted_chunks = []
        
        for raw in raw_chunks:
            if not raw.strip(): continue
            
            title = "Section"
            content = raw
            
            title_match = re.search(r'TITLE:\s*(.*?)(?:\n|$)', raw, re.IGNORECASE)
            if title_match: title = title_match.group(1).strip()
            
            content_match = re.search(r'CONTENT:\s*(.*)', raw, re.IGNORECASE | re.DOTALL)
            if content_match: content = content_match.group(1).strip()
            else: content = re.sub(r'TITLE:.*?\n', '', raw, flags=re.IGNORECASE).strip()

            if len(content) < 20: continue 

            content_with_header = (
                f"DOCUMENT: {source_doc}\n"
                f"SECTION: {title}\n"
                f"ID: {current_chunk_index}\n"
                f"CONTENT:\n{content}"
            )
            
            formatted_chunks.append({
                "source": source_doc,
                "chunk_index": current_chunk_index,
                "text": content_with_header,
                "metadata": {
                    "source": source_doc,
                    "title": title,
                    "original_text": content
                }
            })
            current_chunk_index += 1
            
        return formatted_chunks, remainder

    except Exception as e:
        print(f"  [Error] Processing failed for buffer: {e}")
        return [], text_buffer

def process_document_pipeline(filepath, filename):
    print(f"\nProcessing: {filename}...")
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        print(f"Error opening PDF {filename}: {e}")
        return []
    
    # 1. CLEANING
    tasks = []
    for i, page in enumerate(doc):
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        tasks.append((new_doc.tobytes(), i))
        new_doc.close()
    doc.close()

    cleaned_pages = {}
    print(f"  > Cleaning {len(tasks)} pages...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(clean_page_content_vision, [t[0] for t in tasks]), total=len(tasks), unit="pg"))
        for i, text in enumerate(results):
            if text: cleaned_pages[i] = text

    full_text_stream = ""
    for i in range(len(tasks)):
        if i in cleaned_pages:
            full_text_stream += cleaned_pages[i] + "\n\n"

    # 2. CHUNKING
    print(f"  > Semantic Chunking ({len(full_text_stream)} chars)...")
    final_chunks = []
    chunk_idx_counter = 0 # Questo verrà aggiustato dopo nel main
    buffer = ""
    words = full_text_stream.split()
    current_word_idx = 0
    
    while current_word_idx < len(words):
        while len(buffer) < LLM_BUFFER_SIZE and current_word_idx < len(words):
            buffer += words[current_word_idx] + " "
            current_word_idx += 1
        
        is_last_batch = (current_word_idx >= len(words))
        new_chunks, remainder = llm_semantic_split(buffer, filename, chunk_idx_counter)
        
        if new_chunks:
            final_chunks.extend(new_chunks)
            chunk_idx_counter += len(new_chunks)
        
        if is_last_batch:
            if remainder.strip():
                final_chunks.append({
                    "source": filename,
                    "chunk_index": chunk_idx_counter,
                    "text": f"DOCUMENT: {filename}\nSECTION: Final\nCONTENT:\n{remainder}",
                    "metadata": {"source": filename, "original_text": remainder}
                })
            break
        else:
            buffer = remainder + " " 
            if len(remainder) > len(buffer) * 0.9:
                cut = len(buffer) // 2
                final_chunks.append({
                    "source": filename,
                    "chunk_index": chunk_idx_counter,
                    "text": f"DOCUMENT: {filename}\nSECTION: Raw\nCONTENT:\n{buffer[:cut]}",
                    "metadata": {"source": filename, "original_text": buffer[:cut]}
                })
                chunk_idx_counter += 1
                buffer = buffer[cut:]

    print(f"  > Created {len(final_chunks)} semantic chunks.")
    return final_chunks

def main():
    if not os.path.exists(SOURCE_DOCS_DIR):
        print(f"Directory '{SOURCE_DOCS_DIR}' not found.")
        return

    # 1. Carica chunks esistenti
    existing_chunks = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
        print(f"Loaded {len(existing_chunks)} existing chunks from {OUTPUT_FILE}")
    else:
        print("No existing chunk file found. Starting fresh.")

    # 2. Processa SOLO i file mancanti/corrotti
    new_chunks = []
    
    for filename in FILES_TO_REPROCESS:
        filepath = os.path.join(SOURCE_DOCS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: File {filename} not found in documents folder!")
            continue
            
        doc_chunks = process_document_pipeline(filepath, filename)
        new_chunks.extend(doc_chunks)

    # 3. Merge e Salvataggio
    if new_chunks:
        # Aggiusta gli ID progressivi dei nuovi chunk per non sovrapporsi (opzionale ma pulito)
        max_id = 0
        if existing_chunks:
            # Trova l'ID più alto esistente (se usi ID numerici globali, altrimenti ignora)
            pass 

        total_chunks = existing_chunks + new_chunks
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(total_chunks, f, ensure_ascii=False, indent=4)
        
        print(f"\nSUCCESS! Appended {len(new_chunks)} new chunks.")
        print(f"Total chunks in file: {len(total_chunks)}")
    else:
        print("\nNo new chunks were generated.")

if __name__ == "__main__":
    main()