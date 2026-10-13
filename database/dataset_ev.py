import os
import json
import pandas as pd
import boto3
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)

# Load .env from grandparent directory
dotenv_path = os.path.join(GRANDPARENT_DIR, ".env")
load_dotenv(dotenv_path)

# Paths
INPUT_FILE = os.path.join(CURRENT_DIR, "0_q&a_dataset.json")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "1_q&a_evaluated_test.jsonl")

# AWS Bedrock Config
JUDGE_INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_CLAUDE_OPUS_4_5")
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

if not JUDGE_INFERENCE_PROFILE:
    print("WARNING: INFERENCE_PROFILE not found in .env.")

JUDGE_CONFIG = {
    "maxTokens": 1024,
    "temperature": 0.5
}

# =================================================

class ThesisEvaluator:
    def __init__(self):
        print("Initializing Evaluator...")
        self.bedrock = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

    def call_bedrock_judge(self, prompt):
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        try:
            response = self.bedrock.converse(
                modelId=JUDGE_INFERENCE_PROFILE,
                messages=messages,
                inferenceConfig=JUDGE_CONFIG
            )
            return response['output']['message']['content'][0]['text'].strip()
        except Exception as e:
            print(f"Judge Error: {e}")
            return None

    def evaluate_single_answer(self, question, answer):
        prompt = f"""
        Agisci come un "Data Quality Expert" specializzato nell'addestramento di LLM per il settore legale-energetico (Energy Law).
        Il tuo compito è valutare un esempio di training (User Question, Assistant Answer) e assegnare un punteggio di qualità da 0 a 10 per un task di Fine-Tuning.

        L'obiettivo è creare un assistente per EDISON che risponda come un CONSULENTE TECNICO OPERATIVO.

        CRITERI DI VALUTAZIONE:

        1. STILE E TONO:
        - Penalizza frasi fatte ("La disciplina riflette...", "La ratio...").
        - Penalizza tono filosofico/astratto.
        - Premia tono pratico ("Operativamente...", "Ai sensi dell'art. X...").
        - Premia tecnicismi corretti ("capacity hoarding", "remit", "balancing").

        2. STRUTTURA:
        - Basso se è una definizione da dizionario.
        - Alto se fornisce una linea guida o parere operativo.

        3. COMPLETENZA:
        - Valuta coerenza, esaustività e formattazione.

        ISTRUZIONI CRITICHE SUL PUNTEGGIO (Per ottenere massima precisione):
        - Valuta su una SCALA CONTINUA.
        - Usa i decimali per riflettere piccole sfumature.

        OUTPUT RICHIESTO:
        Rispondi ESCLUSIVAMENTE con un oggetto JSON in questo formato (nessun altro testo):
        {{
        "score": <numero float da 0.0 a 10.0, usa una sola cifra decimale>
        }}
        """
        final_prompt = f"{prompt}\n\nQUESTION: {question}\nANSWER: {answer}"
        
        response_text = self.call_bedrock_judge(final_prompt)
        if not response_text: return 0.0

        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)
            return float(data.get("score", 0.0))
        except:
            return 0.0
        
    def process_item(self, idx, item):
        try:
            messages = item.get("messages", [])
            user = next((m["content"] for m in messages if m["role"] == "user"), None)
            asst = next((m["content"] for m in messages if m["role"] == "assistant"), None)

            if user and asst:
                score = self.evaluate_single_answer(user, asst)
                item["score"] = score
            else:
                item["score"] = 0.0
            
            return idx, item
        except Exception as e:
            print(f"Error item {idx}: {e}")
            item["score"] = 0.0
            return idx, item

    def run_evaluation(self):
        if not os.path.exists(INPUT_FILE):
            print(f"File not found: {INPUT_FILE}")
            return

        print(f"Loading {INPUT_FILE}...")
        data = []
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # Gestione ibrida: se inizia con [ è una lista JSON, altrimenti JSONL
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
        
        print(f"Evaluating {len(data)} items...")
        results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.process_item, i, item) for i, item in enumerate(data)]
            for future in tqdm(as_completed(futures), total=len(data)):
                idx, res = future.result()
                results.append(res)

        print(f"Saving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    evaluator = ThesisEvaluator()
    evaluator.run_evaluation()