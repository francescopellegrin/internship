import json
import numpy as np
import os

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)

# Paths
INPUT_FILE = os.path.join(CURRENT_DIR, "1_q&a_evaluated_test.jsonl")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "2_q&a_cleaned.jsonl")

def clean_thesis_dataset(input_file, output_file):
    """
    Pulisce il dataset rimuovendo solo i punteggi bassi (outlier negativi)
    usando il metodo IQR. Ottimizzato per una distribuzione sbilanciata verso l'alto.
    """
    
    # 1. Caricamento Dati
    if not os.path.exists(input_file):
        print(f"Errore: Il file {input_file} non esiste.")
        return

    data = []
    scores = []
    
    print(f"Lettura file: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'score' in entry and isinstance(entry['score'], (int, float)):
                    data.append(entry)
                    scores.append(entry['score'])
            except json.JSONDecodeError:
                continue

    if not scores:
        print("Dataset vuoto o senza campo 'score'.")
        return

    # 2. Calcolo Statistiche (IQR)
    scores_np = np.array(scores)
    
    q1 = np.percentile(scores_np, 25)
    q3 = np.percentile(scores_np, 75)
    iqr = q3 - q1
    
    # Moltiplicatore 1.5 è lo standard statistico per gli outlier
    multiplier = 1
    lower_bound = q1 - (multiplier * iqr)
    
    print("-" * 40)
    print(f"ANALISI STATISTICA SUI {len(scores)} ESEMPI")
    print("-" * 40)
    print(f"Q1 (25%ile): {q1:.4f}")
    print(f"Q3 (75%ile): {q3:.4f}")
    print(f"IQR:        {iqr:.4f}")
    print(f"SOGLIA DI TAGLIO (Lower Bound): {lower_bound:.4f}")
    print("-" * 40)

    # 3. Filtraggio
    kept_data = []
    removed_count = 0
    removed_distribution = {} # Per vedere cosa stiamo buttando

    for entry in data:
        score = entry['score']
        
        # Manteniamo tutto ciò che è sopra il lower_bound
        if score > lower_bound:
            kept_data.append(entry)
        else:
            removed_count += 1
            # Tracciamo quali voti stiamo buttando per conferma
            score_int = int(score)
            removed_distribution[score_int] = removed_distribution.get(score_int, 0) + 1

    # 4. Salvataggio
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in kept_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"PULIZIA COMPLETATA")
    print(f"Esempi rimossi: {removed_count}")
    print(f"Distribuzione rimossi (per range): {dict(sorted(removed_distribution.items()))}")
    print(f"Esempi validi salvati: {len(kept_data)}")
    print(f"File salvato in: {output_file}")

clean_thesis_dataset(INPUT_FILE, OUTPUT_FILE)

"""
----------------------------------------
ANALISI STATISTICA SUI 1008 ESEMPI
----------------------------------------
Q1 (25%ile): 6.8000
Q3 (75%ile): 7.8000
IQR:        1.0000
SOGLIA DI TAGLIO (Lower Bound): 5.8000
----------------------------------------
PULIZIA COMPLETATA
Esempi rimossi: 52
Distribuzione rimossi (per range): {3: 2, 4: 4, 5: 46}
Esempi validi salvati: 956
File salvato in: /home/ec2-user/tesi/draft/2_q&a_cleaned.jsonl
"""


""" --- RISULTATI ---
----------------------------------------
ANALISI STATISTICA SUI 1008 ESEMPI
----------------------------------------
Q1 (25%ile): 6.8700
Q3 (75%ile): 7.6800
IQR:        0.8100
SOGLIA DI TAGLIO (Lower Bound): 6.0600
----------------------------------------
PULIZIA COMPLETATA
Esempi rimossi: 54
Distribuzione rimossi (per range): {2: 1, 3: 1, 4: 3, 5: 49}
Esempi validi salvati: 954
"""

