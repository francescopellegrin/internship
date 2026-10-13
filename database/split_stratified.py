import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(CURRENT_DIR, "3_q&a_indexed.jsonl") # File indicizzato

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# =================================================

def stratified_split():
    print(f"Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    df = pd.read_json(INPUT_FILE, lines=True)
    total_size = len(df)
    print(f"Total dataset size: {total_size}")

    # 1. Creazione Bin per stratificazione
    # Dividiamo gli score in 5 fasce (quintili) per garantire che ogni split 
    # abbia la stessa proporzione di voti bassi, medi e alti.
    # duplicates='drop' gestisce il caso in cui molti voti sono identici.
    df['stratify_group'] = pd.qcut(df['score'], q=5, labels=False, duplicates='drop')
    
    print("Stratification bins created.")

    # 2. Split: Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(VAL_RATIO + TEST_RATIO), 
        random_state=42, 
        stratify=df['stratify_group'] # Assicura distribuzione bilanciata
    )

    # 3. Split: Temp (30%) -> Val (15%) e Test (15%)
    # La proporzione relativa è 0.5 (metà del rimanente va a test)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['stratify_group']
    )

    # Rimuoviamo la colonna di appoggio
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['stratify_group'], inplace=True)

    # Salvataggio
    datasets = {
        'train.jsonl': train_df,
        'validation.jsonl': val_df,
        'test.jsonl': test_df
    }

    print("\nSaving split files...")
    for filename, dataframe in datasets.items():
        out_path = os.path.join(CURRENT_DIR, filename)
        dataframe.to_json(out_path, orient='records', lines=True, force_ascii=False)
        print(f" -> {filename}: {len(dataframe)} items")

if __name__ == "__main__":
    stratified_split()