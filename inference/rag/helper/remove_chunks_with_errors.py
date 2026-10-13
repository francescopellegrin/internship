import json
import os


# --- Configuration & Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_FILE = os.path.join(CURRENT_DIR, "chunks_smart.json")

# LISTA DEI FILE DA RIMUOVERE (Quelli che hanno dato errore nei log precedenti)
# Assicurati che i nomi corrispondano esattamente a quelli nella cartella documents
FILES_TO_CLEAN = [
    "Legge del 30 ottobre 2014.pdf",
    "REMIT Regolamento.pdf",
    "REMIT updated.pdf",          # Controlla se il nome su disco è questo o "5th Edition..."
    "ARERA Sanzione 2022_306.pdf",
    "ARERA Sanzione 2022_307.pdf"
]

def clean_chunks():
    if not os.path.exists(CHUNKS_FILE):
        print(f"Errore: Il file {CHUNKS_FILE} non esiste.")
        return

    print(f"Caricamento {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    initial_count = len(data)
    print(f"Chunk totali iniziali: {initial_count}")

    # Filtra mantenendo solo i chunk che NON sono nella lista nera
    # Usiamo lower() per evitare problemi di maiuscole/minuscole
    files_lower = [f.lower() for f in FILES_TO_CLEAN]
    
    clean_data = [
        chunk for chunk in data 
        if chunk.get("source", "").strip() not in FILES_TO_CLEAN
        # Controllo extra per sicurezza sui nomi (case insensitive)
        and chunk.get("source", "").strip().lower() not in files_lower
    ]

    removed_count = initial_count - len(clean_data)
    
    # Salvataggio
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=4)

    print(f"--- PULIZIA COMPLETATA ---")
    print(f"Rimossi {removed_count} chunk corrotti o parziali.")
    print(f"Chunk validi rimanenti: {len(clean_data)}")
    print(f"File sovrascritto: {CHUNKS_FILE}")

if __name__ == "__main__":
    clean_chunks()