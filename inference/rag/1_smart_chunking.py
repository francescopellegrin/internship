import os
import json
import boto3
import re
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

# --- Configuration & Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

SOURCE_DOCS_DIR = os.path.join(CURRENT_DIR, "documents")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "chunks_smart.json")

# Usa un modello veloce ed economico ma smart (Haiku è ottimo, Llama 70b anche)
CHUNKING_MODEL_ID = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5", "anthropic.claude-3-haiku-20240307-v1:0") 
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

# Dimensione del buffer da passare all'LLM (caratteri)
# 12.000 chars ~ 3000 token. Abbastanza per dare contesto, non troppo per i costi.
LLM_BUFFER_SIZE = 12000 
MAX_WORKERS = 5

# --- Bedrock Client ---
session = boto3.Session(region_name=REGION_NAME)
client = session.client("bedrock-runtime")

def clean_page_content_vision(page_bytes):
    """
    Step 1: OCR & Cleaning visuale (rimuove header/footer/numeri pagina)
    """
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
    Versione Robustificata con Self-Correction per errori JSON.
    """
    system_prompt = (
        "You are a Semantic Chunker for RAG systems. Split the text into logical, self-contained chunks.\n"
        "RULES:\n"
        "1. JSON Output ONLY. No intro/outro text.\n"
        "2. Structure: { 'chunks': [ { 'title': '...', 'content': '...' } ], 'remainder': '...' }\n"
        "3. ESCAPE ALL QUOTES inside content strings carefully.\n"
        "4. If text ends abruptly, put that part in 'remainder'.\n"
        "5. Avoid creating chunks smaller than 200 characters unless it's a standalone definition."
    )

    # Escape minimale per input
    safe_text = text_buffer.replace('"', "'")
    user_msg = f"Input Text ({len(text_buffer)} chars):\n<text>\n{safe_text}\n</text>\n\nGenerate JSON."

    try:
        response = client.converse(
            modelId=CHUNKING_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 4096}
        )
        
        output_text = response['output']['message']['content'][0]['text']
        
        # Tentativo di pulizia JSON (trova la prima { e l'ultima })
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON brackets found")
            
        json_str = output_text[start:end]
        data = json.loads(json_str) # Qui può fallire
        
        # Parsing riuscito -> Creazione oggetti
        formatted_chunks = []
        for item in data.get("chunks", []):
            content_str = item.get('content', '').strip()
            if len(content_str) < 20: continue # Salta rumore

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
        # SELF-CORRECTION: Se fallisce, riprova chiedendo al modello di fixare
        if retry_count < 2:
            print(f"  [JSON Error] Retrying with correction (Attempt {retry_count+1})...")
            # Chiamata ricorsiva semplice o logica di fix (qui semplice retry per robustezza)
            # Spesso basta riprovare perché la temperature non è mai perfettamente zero nell'hardware
            return llm_semantic_split(text_buffer, source_doc, current_chunk_index, retry_count + 1)
        else:
            print(f"  [JSON Failure] Skipping batch for {source_doc} after retries.")
            # FALLBACK: Restituisci tutto come remainder per non perdere dati
            return [], text_buffer
            
    except Exception as e:
        print(f"  [General Error] {e}")
        return [], text_buffer

def process_document_pipeline(filepath, filename):
    """
    Orchestra: PDF -> Immagini -> Testo Pulito -> Buffer -> LLM Chunking
    """
    print(f"\nProcessing: {filename}...")
    doc = fitz.open(filepath)
    
    # 1. CLEANING PARALLELO (Vision)
    tasks = []
    for i, page in enumerate(doc):
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        tasks.append((new_doc.tobytes(), i))
        new_doc.close()
    doc.close()

    cleaned_pages = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(clean_page_content_vision, [t[0] for t in tasks]), total=len(tasks), desc="  Cleaning Pages", unit="pg"))
        for i, text in enumerate(results):
            if text: cleaned_pages[i] = text

    # Ricostruiamo il testo completo in ordine
    full_text_stream = ""
    for i in range(len(tasks)):
        if i in cleaned_pages:
            full_text_stream += cleaned_pages[i] + "\n\n"

    # 2. SEMANTIC CHUNKING (LLM)
    print(f"  > Semantic Chunking ({len(full_text_stream)} chars)...")
    
    final_chunks = []
    chunk_idx_counter = 0
    buffer = ""
    
    # Dividiamo il testo in finestre per l'LLM
    # Non usiamo split fissi, ma riempiamo il buffer fino a LLM_BUFFER_SIZE
    # e lasciamo che l'LLM gestisca il resto.
    
    words = full_text_stream.split()
    current_word_idx = 0
    
    while current_word_idx < len(words):
        # Riempi buffer
        while len(buffer) < LLM_BUFFER_SIZE and current_word_idx < len(words):
            buffer += words[current_word_idx] + " "
            current_word_idx += 1
        
        # Se siamo alla fine, flagghiamo per forzare il chunking anche del residuo
        is_last_batch = (current_word_idx >= len(words))
        
        # Chiamata LLM
        new_chunks, remainder = llm_semantic_split(buffer, filename, chunk_idx_counter)
        
        if new_chunks:
            final_chunks.extend(new_chunks)
            chunk_idx_counter += len(new_chunks)
        
        # Gestione Avanzamento
        if is_last_batch:
            # Se è rimasto qualcosa alla fine assoluta, fanne un chunk forzato
            if remainder.strip():
                final_chunks.append({
                    "source": filename,
                    "chunk_index": chunk_idx_counter,
                    "text": f"DOCUMENT: {filename}\nSECTION: Final Segment\nCONTENT:\n{remainder}",
                    "metadata": {"source": filename, "original_text": remainder}
                })
            break
        else:
            # Il "remainder" diventa l'inizio del prossimo buffer
            # Questo garantisce che una frase tagliata a metà venga completata nel prossimo giro
            buffer = remainder + " " 
            
            # Safety: Se l'LLM non ha tagliato nulla (remainder == buffer), 
            # forziamo un taglio meccanico per evitare loop infiniti
            if len(remainder) > len(buffer) * 0.9:
                # Taglia il 50% del buffer e vai avanti
                cut_point = len(buffer) // 2
                processed_part = buffer[:cut_point]
                buffer = buffer[cut_point:]
                # Aggiungi il pezzo processato come chunk "grezzo" (fallback estremo)
                final_chunks.append({
                    "source": filename,
                    "chunk_index": chunk_idx_counter,
                    "text": f"DOCUMENT: {filename}\nSECTION: Segment {chunk_idx_counter}\nCONTENT:\n{processed_part}",
                    "metadata": {"source": filename, "original_text": processed_part}
                })
                chunk_idx_counter += 1

    print(f"  > Created {len(final_chunks)} semantic chunks.")
    return final_chunks

def main():
    if not os.path.exists(SOURCE_DOCS_DIR):
        print(f"Directory '{SOURCE_DOCS_DIR}' not found.")
        return

    all_chunks = []
    pdf_files = [f for f in os.listdir(SOURCE_DOCS_DIR) if f.lower().endswith('.pdf')]
    
    for filename in pdf_files:
        filepath = os.path.join(SOURCE_DOCS_DIR, filename)
        doc_chunks = process_document_pipeline(filepath, filename)
        all_chunks.extend(doc_chunks)

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print(f"\nProcessing complete. {len(all_chunks)} total chunks saved.")

if __name__ == "__main__":
    main()