import json
import random
import os

# CONFIGURAZIONE PATH PER EC2
# ---------------------------------------------------------
# Rileva automaticamente la directory corrente come base assoluta
# os.getcwd() restituisce il percorso assoluto della cartella da cui lanci lo script
# base_dir = os.path.abspath(os.getcwd())

# Se preferisci fissare una cartella specifica sull'EC2, decommenta e modifica la riga sotto:
base_dir = "/home/ec2-user/tesi/draft" 

input_filename = "q&a_dataset.json"
input_path = os.path.join(base_dir, input_filename)

output_files_config = {
    'train': "trainC.jsonl",
    'validation': "validationC.jsonl",
    'test': "testC.jsonl"
}
# ---------------------------------------------------------

# Seed per riproducibilità
random.seed(42)

def split_dataset():
    print(f"Ambiente rilevato: EC2 / Linux Standard")
    print(f"Base directory (path assoluto): {base_dir}")
    print(f"File input atteso: {input_path}")

    # Carica il dataset
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n[ERRORE] File non trovato: {input_path}")
        print("Assicurati di aver caricato il file 'q&a_testset.json' in questa cartella.")
        return

    if not isinstance(data, list):
        print("[ERRORE] Il formato del JSON non è una lista.")
        return

    total_size = len(data)
    print(f"Totale esempi caricati: {total_size}")

    # Mischia i dati
    random.shuffle(data)

    train_end = int(0.8 * total_size)
    val_end = train_end + int(0.1 * total_size)

    datasets = {
        'train': data[:train_end],
        'validation': data[train_end:val_end],
        'test': data[val_end:]
    }

    # Salvataggio
    print("\nSalvataggio file...")
    for key, content in datasets.items():
        # Costruzione path assoluto di output
        output_abs_path = os.path.join(base_dir, output_files_config[key])
        
        with open(output_abs_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4, ensure_ascii=False)
        
        print(f" -> {output_abs_path} ({len(content)} items)")

if __name__ == "__main__":
    split_dataset()
    #A = (70,15,15) --> (357, 76, 77)
    #B = (60,20,20) --> (306, 102, 102)
    #C = (80,10,10) --> (408, 51, 51)