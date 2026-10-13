import json
import pandas as pd
import numpy as np
import os

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Paths
INPUT_FILE = os.path.join(CURRENT_DIR, "2_q&a_cleaned.jsonl")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "3_q&a_indexed.jsonl")
# =================================================

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return pd.DataFrame(data)

def add_index_and_sort(df):
    print("--- SORTING & INDEXING ---")
    
    # 1. Ordina per score CRESCENTE (dal più basso al più alto)
    df_sorted = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # 2. Aggiunge colonna index (0, 1, 2, ...)
    df_sorted['index'] = df_sorted.index
    
    # 3. Riordina colonne: Index come primo campo, Score come ultimo
    cols = list(df_sorted.columns)
    if 'index' in cols: cols.remove('index')
    if 'score' in cols: cols.remove('score')
    
    final_cols = ['index'] + cols + ['score']
    df_final = df_sorted[final_cols]
    
    print(f"Sorted {len(df_final)} items by score (Ascending).")
    print(f"First Item Score: {df_final.iloc[0]['score']} (Low)")
    print(f"Last Item Score: {df_final.iloc[-1]['score']} (High)")
    
    return df_final

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file missing: {INPUT_FILE}")
        return

    df = load_data(INPUT_FILE)
    df_final = add_index_and_sort(df)
    
    print(f"Saving indexed file to {OUTPUT_FILE}")
    df_final.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    main()