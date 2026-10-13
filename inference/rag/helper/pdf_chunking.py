import os
import json
import boto3
import random
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
OUTPUT_FILE = os.path.join(CURRENT_DIR, "chunks.json")

# Model Settings
CLEANING_MODEL_ID = os.getenv("INFERENCE_PROFILE_CLAUDE_HAIKU_4_5") 
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

# Chunking Settings for Legal Text (Characters, not tokens)
# 2500 chars ~ 600 tokens (Better for RAG context)
TARGET_CHUNK_SIZE = 2500  
# 500 chars overlap ~ 100-120 tokens (Ensures boundary sentences are preserved)
OVERLAP_SIZE = 500  
TEST_DOC_LIMIT = 3
MAX_WORKERS = 5  # Parallel pages to process at once

# --- Bedrock Client ---
session = boto3.Session(region_name=REGION_NAME)
client = session.client("bedrock-runtime")

def clean_page_content_vision(page_bytes):
    """
    Sends the PDF page bytes directly to Bedrock.
    """
    system_prompt = (
        "You are a specialized data cleaning assistant for legal documents in the energy sector. "
        "Your task is to extract the core legal text from the provided PDF page content.\n"
        "Rules:\n"
        "1. IGNORE headers, page numbers, and logos. IGNORE boilerplate footers (e.g. 'Confidential').\n"
        "2. Remove Table of Contents (indices) entirely.\n"
        "3. Remove boilerplate disclaimers (e.g., 'This document is property of...').\n"
        "4. Fix hyphenated words broken across lines (e.g., 'regu- lation' -> 'regulation').\n"
        "5. Preserve the exact legal wording, article numbers, and structure.\n"
        "6. If a page contains ONLY a Table of Contents or ONLY legal boilerplate, return the string 'SKIP'.\n"
        "7. HANDLING FOOTNOTES: Do not leave footnotes at the bottom. "
        "   If you see a footnote reference (e.g., a number '1' or symbol '*'), find the corresponding text at the bottom of the page"
        "   and insert it IMMEDIATELY after the sentence it refers to, enclosed in double brackets like [[Footnote: ...]].\n"
        "8. Output ONLY the cleaned text. Do not add conversational fillers."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "document": {
                        "name": "document_page",
                        "format": "pdf", 
                        "source": {
                            "bytes": page_bytes
                        }
                    }
                },
                {
                    "text": "Please transcribe and clean this page according to the system rules."
                }
            ]
        }
    ]

    try:
        response = client.converse(
            modelId=CLEANING_MODEL_ID,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 4000}
        )
        return response['output']['message']['content'][0]['text'].strip()

    except Exception as e:
        # Return None so we can handle the failure gracefully
        return None

def process_single_page(args):
    """
    Worker function for parallel processing.
    """
    page_bytes, page_num = args
    text = clean_page_content_vision(page_bytes)
    return (page_num, text)

def chunk_text_smart(text, source_doc):
    """
    Revised chunking with SLIDING WINDOW overlap to prevent context loss.
    """
    chunks = []
    # This regex keeps the separator (space) in the list
    words = re.split(r'(\s+)', text)
    
    # Filter out empty strings if any
    words = [w for w in words if w]
    
    chunk_index = 0
    start_idx = 0
    
    while start_idx < len(words):
        current_chunk_words = []
        current_char_count = 0
        end_idx = start_idx
        
        # --- PHASE 1: Fill the buffer to target size ---
        while end_idx < len(words):
            word = words[end_idx]
            word_len = len(word)
            
            # If adding this word exceeds limit significantly, check for break
            if current_char_count + word_len > TARGET_CHUNK_SIZE:
                # We are at the limit. 
                # OPTIMIZATION: Try to find a sentence break (.) or paragraph break (\n)
                # within the last 10% of the buffer to avoid cutting mid-sentence.
                break
            
            current_chunk_words.append(word)
            current_char_count += word_len
            end_idx += 1
        
        # --- PHASE 2: Refine the Cut (Backtracking) ---
        # If we didn't reach the end of the text, try to find a natural break point
        # looking backwards from end_idx
        if end_idx < len(words):
            lookback_limit = 50  # Check last 50 words for a period
            best_break = -1
            
            # Look backwards in the current accumulation
            for i in range(len(current_chunk_words)-1, max(0, len(current_chunk_words)-lookback_limit), -1):
                w = current_chunk_words[i]
                # Check for paragraph break
                if "\n\n" in w:
                    best_break = i + 1
                    break
                # Check for sentence break (period followed by space usually)
                if "." in w and len(w) > 1 and not w.lower().endswith(("mr.", "dr.", "art.", "d.lgs.", "n.", "comma", "lett.", "reg.")):
                    best_break = i + 1 # Include the period in this chunk
                    break
            
            # If we found a semantic break, shorten the chunk to that point
            if best_break != -1:
                # Adjust end_idx in the main 'words' list
                diff = len(current_chunk_words) - best_break
                end_idx -= diff
                current_chunk_words = current_chunk_words[:best_break]

        # --- PHASE 3: Save Chunk ---
        chunk_str = "".join(current_chunk_words).strip()
        
        # Create a rich metadata string to help the LLM explicitly
        # "Context: Document [Name], Chunk [N]"
        content_with_header = f"DOCUMENT: {source_doc}\nSECTION ID: {chunk_index}\nCONTENT:\n{chunk_str}"

        if len(chunk_str) > 50: # Ignore tiny garbage chunks
            chunks.append({
                "source": source_doc,
                "chunk_index": chunk_index,
                "text": content_with_header, # Use the header version for embedding/retrieval
                "metadata": {
                    "source": source_doc, 
                    "chunk_index": chunk_index,
                    "original_text": chunk_str
                }
            })
            chunk_index += 1
        
        # --- PHASE 4: Calculate Next Start (The Overlap Logic) ---
        # We need to move start_idx forward, but not all the way to end_idx.
        # We want the NEW start_idx to be `end_idx` minus enough words to fill OVERLAP_SIZE.
        
        if end_idx == len(words):
            break # We are done
            
        overlap_char_count = 0
        new_start_idx = end_idx
        
        # Walk backwards from end_idx to find where the overlap should start
        while new_start_idx > start_idx and overlap_char_count < OVERLAP_SIZE:
            new_start_idx -= 1
            overlap_char_count += len(words[new_start_idx])
            
        # Safety: Ensure we always advance at least one word to prevent infinite loops
        if new_start_idx == start_idx:
            new_start_idx += 1
            
        start_idx = new_start_idx

    return chunks

def process_documents():
    all_chunks = []
    
    if not os.path.exists(SOURCE_DOCS_DIR):
        print(f"Directory '{SOURCE_DOCS_DIR}' not found.")
        return

    # Get ALL PDF files
    pdf_files = [f for f in os.listdir(SOURCE_DOCS_DIR) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF documents in total.")

    for filename in pdf_files:
        filepath = os.path.join(SOURCE_DOCS_DIR, filename)
        print(f"\nProcessing: {filename}...")
        
        try:
            doc = fitz.open(filepath)
            tasks = []
            
            # 1. Prepare Pages (Extract bytes)
            # We do this in main thread because fitz objects aren't pickleable easily
            for i, page in enumerate(doc):
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=i, to_page=i)
                page_bytes = new_doc.tobytes()
                new_doc.close()
                tasks.append((page_bytes, i+1))
            
            doc.close()

            # 2. Process Pages in Parallel
            cleaned_pages_map = {}
            print(f"  > Cleaning {len(tasks)} pages with {MAX_WORKERS} workers...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Map returns results in order if using map, but we used submit pattern implicitly via args
                results = list(tqdm(
                    executor.map(process_single_page, tasks), 
                    total=len(tasks), 
                    unit="page"
                ))
                
                for page_num, text in results:
                    if text and text != "SKIP" and len(text) > 20:
                        cleaned_pages_map[page_num] = text

            # 3. Reconstruct Order
            ordered_text = []
            # We iterate 1..N to ensure perfect order regardless of when threads finished
            for i in range(1, len(tasks) + 1):
                if i in cleaned_pages_map:
                    ordered_text.append(cleaned_pages_map[i])
            
            full_doc_text = "\n\n".join(ordered_text)

            # 4. Smart Chunking
            if full_doc_text:
                print(f"  > Smart Chunking ({len(full_doc_text)} chars)...")
                doc_chunks = chunk_text_smart(full_doc_text, filename)
                all_chunks.extend(doc_chunks)
                print(f"  > Created {len(doc_chunks)} chunks.")
            else:
                print("  > No valid text extracted.")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    
    print(f"\nProcessing complete. {len(all_chunks)} chunks saved.")

if __name__ == "__main__":
    process_documents()