import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Files
INPUT_FILE = os.path.join(CURRENT_DIR, "results/qwen3-32b.json")

# La stringa esatta che il modello usa quando non sa rispondere
# (Dal tuo esempio sembra essere questa, inclusa la newline eventuale)
NOT_FOUND_MARKER = "RISPOSTA_NON_TROVATA"

def is_refusal(text):
    """Restituisce True se il testo contiene un marker di rifiuto."""
    if not text: return True
    marker = NOT_FOUND_MARKER
    if marker in text:
        return True
    return False

def analyze_performance():
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: Il file {INPUT_FILE} non esiste.")
        return

    print(f"--- ANALISI PERFORMANCE: {INPUT_FILE} ---\n")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Contatori per la Confusion Matrix
    TP = 0  # Grounded=True  & Risposta Data
    TN = 0  # Grounded=False & Rifiuto ([RISPOSTA_NON_TROVATA])
    FP = 0  # Grounded=False & Risposta Data (Allucinazione)
    FN = 0  # Grounded=True  & Rifiuto (Errore di Retrieval o Eccessiva Cautela)

    total_responses = 0
    
    # Liste per debug (salviamo gli ID delle domande problematiche)
    hallucinations = []   # FP
    missed_answers = []   # FN

    for entry in data:
        # Se manca il campo is_grounded, non possiamo valutare
        if "is_grounded" not in entry:
            continue

        ground_truth = entry["is_grounded"]
        responses = entry.get("generated_responses", [])

        # Analizziamo ogni risposta generata (es. 5 per domanda)
        for resp in responses:
            total_responses += 1
            refused = is_refusal(resp)

            if ground_truth and not refused:
                TP += 1
            elif not ground_truth and refused:
                TN += 1
            elif not ground_truth and not refused:
                FP += 1
                if entry["id"] not in hallucinations: hallucinations.append(entry["id"])
            elif ground_truth and refused:
                FN += 1
                if entry["id"] not in missed_answers: missed_answers.append(entry["id"])

    # Calcolo Metriche
    if total_responses == 0:
        print("Nessuna risposta trovata nel file.")
        return

    accuracy = (TP + TN) / total_responses
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    hallucination_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

    print(f"Totale Risposte Analizzate: {total_responses}")
    print("-" * 50)
    print(f"{'':<20} | {'Grounded (True)':<15} | {'Ungrounded (False)':<15}")
    print("-" * 50)
    print(f"{'Model Answered':<20} | TP: {TP:<11} | FP: {FP:<11} (Hallucinations)")
    print(f"{'Model Refused':<20} | FN: {FN:<11} | TN: {TN:<11}")
    print("-" * 50)

    print("\nMETRICHE:")
    print(f"1. Accuracy:             {accuracy:.2%}")
    print(f"2. Recall (Completeness):  {recall:.2%}  (Capacità di rispondere quando serve)")
    print(f"3. Precision (Trust):      {precision:.2%}  (Affidabilità delle risposte date)")
    print(f"4. Hallucination Rate:     {hallucination_rate:.2%}  (Quanto inventa su domande senza risposta)")

    print("\n--- DEBUG (Primi 10 ID) ---")
    print(f"FP (Allucinazioni): {hallucinations}")
    print(f"FN (Risposte Perse): {missed_answers}")

if __name__ == "__main__":
    analyze_performance()