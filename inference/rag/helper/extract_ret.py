import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_BEST_RUN = os.path.join(CURRENT_DIR, "results/gemma-27b_2.json") 
OUTPUT_FILE = os.path.join(CURRENT_DIR, "retrieval_frozen_best.json")

def freeze_retrieval():
    target_file = INPUT_BEST_RUN

    if not os.path.exists(target_file):
        # Fallback se il file è nella root (basname)
        fallback_path = os.path.basename(target_file)
        if os.path.exists(fallback_path):
            target_file = fallback_path
        else:
            print(f"ERRORE: Non trovo il file né in {target_file} né in {fallback_path}")
            # Stampa la directory corrente per debug
            print(f"Current Working Directory: {os.getcwd()}")
            return

    print(f"Estrazione del contesto vincente da: {INPUT_BEST_RUN}...")

    with open(INPUT_BEST_RUN, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frozen_dataset = []
    
    for entry in data:
        # Creiamo l'oggetto standard per lo script 4b
        frozen_entry = {
            "id": entry.get("id"),
            "original_question": entry.get("original_question"),
            "gold_answer": entry.get("gold_answer", ""),
            "score": entry.get("score"),
            "is_grounded": entry.get("is_grounded", True),
            "new_query": entry.get("new_query", ""), 
            "retrieved_context": entry.get("retrieved_context", []),
        }
        frozen_dataset.append(frozen_entry)

    # Salvataggio
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(frozen_dataset, f, ensure_ascii=False, indent=4)

    print(f"FATTO! Creato '{OUTPUT_FILE}' con {len(frozen_dataset)} domande.")
    print("Ora usa questo file come input per lo script 4b.")

if __name__ == "__main__":
    freeze_retrieval()