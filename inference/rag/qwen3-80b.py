import os
import json
import boto3
import concurrent.futures
import time
from tqdm import tqdm
from dotenv import load_dotenv
from botocore.config import Config

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# File Paths
INPUT_FILE = os.path.join(CURRENT_DIR, "retrieval_frozen_best.json")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "results/qwen3-80b.json")

# Model & AWS Settings
INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_QWEN3_NEXT_80B")
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")
NUM_ANSWERS = 5

# Metrics Settings
NOT_FOUND_MARKER = "RISPOSTA_NON_TROVATA"

INFERENCE_CONFIG = {
    "maxTokens": 1024,
    "temperature": 0.5,
    "topP": 0.9
}

# Bedrock Client
retry_config = Config(region_name=REGION_NAME, retries={'max_attempts': 8, 'mode': 'adaptive'})
bedrock_client = boto3.client(service_name='bedrock-runtime', config=retry_config)

# ================= INFERENCE FUNCTIONS =================

def generate_rag_response(query, context_list):
    if not query: return "Error"

    # XML Format
    formatted_context = ""
    for i, ctx in enumerate(context_list):
        formatted_context += f"<doc id='{i+1}'>{ctx}</doc>\n"
    
    # Construct System Prompt
    base_system = "You are a senior legal expert in energy regulation for Edison. " \
                  "Your task is to provide comprehensive operational opinions to support the business, relying strictly on the information provided in the context or on known current regulations.\n" \
                  "Use formal and discursive legal language, citing specific articles and regulatory references.\n" \
                  "Where appropriate, enrich the response with plausible practical cases applied to Edison's industrial reality, avoiding the generation of quantitative data not present in the sources. " \
                  "Give the answer in Italian."
    
    rag_system_prompt = (
        f"{base_system}\n\n"
        "RAG INSTRUCTIONS\n"
        "You have access to a set of retrieved documents enclosed in <context> tags.\n"
        "1. Any mention of specific regulations, articles, obligations, or deadlines must be derived EXCLUSIVELY from the <context>.\n"
        "2. Combine information from multiple chunks to answer fully.\n"
        "3. To fulfill the requirement of providing 'plausible practical cases' (as stated above), you are authorized to apply the rules found in the <context> to hypothetical scenarios based on your general knowledge of the energy sector (e.g. power plant maintenance, trading desk operations).\n"
        "4. If the answer is not present in the <context>, you must reply exactly: [RISPOSTA_NON_TROVATA]\n"
        "5. Cite the documents where possible (eg 'Secondo il documento...').\n"
    )

    # Construct User Message
    user_message = (
        f"<context>\n{formatted_context}\n</context>\n\n"
        f"Domanda: {query}"
    )

    # Inference (Converse API)
    for attempt in range(3):
        try:
            response = bedrock_client.converse(
                modelId=INFERENCE_PROFILE,
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                system=[{"text": rag_system_prompt}],
                inferenceConfig=INFERENCE_CONFIG
            )
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            time.sleep(2)
            if attempt == 2: return f"[ERROR] {str(e)}"

def process_entry(entry):
    user_query = entry["original_question"]
    context = entry["retrieved_context"]
    
    responses = []
    # Generazione parallela
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ANSWERS) as executor:
        futures = [
            executor.submit(generate_rag_response, user_query, context) 
            for _ in range(NUM_ANSWERS)
        ]
        for f in concurrent.futures.as_completed(futures):
            responses.append(f.result())

    result_entry = entry.copy()
    result_entry["generated_responses"] = responses
    result_entry["model_profile"] = INFERENCE_PROFILE
    return result_entry

# ================= METRICS FUNCTIONS =================

def is_refusal(text):
    """Restituisce True se il testo contiene un marker di rifiuto."""
    if not text: return True
    if NOT_FOUND_MARKER in text:
        return True
    return False

def analyze_performance(data):
    """Calcola e stampa le metriche basate sulla lista di dati processati."""
    print(f"\n--- ANALISI PERFORMANCE ---\n")
    
    TP = 0  # Grounded=True  & Risposta Data
    TN = 0  # Grounded=False & Rifiuto
    FP = 0  # Grounded=False & Risposta Data (Allucinazione)
    FN = 0  # Grounded=True  & Rifiuto (Errore di Retrieval o Eccessiva Cautela)

    total_responses = 0
    
    hallucinations = []   # FP
    missed_answers = []   # FN

    for entry in data:
        if "is_grounded" not in entry:
            continue

        ground_truth = entry["is_grounded"]
        responses = entry.get("generated_responses", [])

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

    if total_responses == 0:
        print("Nessuna risposta valida trovata o campo 'is_grounded' mancante.")
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
    print(f"2. Recall (Completeness):  {recall:.2%}  (Capacita' di rispondere quando serve)")
    print(f"3. Precision (Trust):      {precision:.2%}  (Affidabilita' delle risposte date)")
    print(f"4. Hallucination Rate:     {hallucination_rate:.2%}  (Quanto inventa su domande senza risposta)")

    print("\n--- DEBUG (Primi 10 ID) ---")
    print(f"FP (Allucinazioni): {hallucinations[:10]}")
    print(f"FN (Risposte Perse): {missed_answers[:10]}")

# ================= MAIN Execution =================

def main():
    print(f"--- RUNNING INFERENCE BENCHMARK ---")
    print(f"Model: {INFERENCE_PROFILE}")
    print(f"Input: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("Run retrieving first!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    
    # Fase 1: Inferenza
    for entry in tqdm(data, desc=f"Testing {INFERENCE_PROFILE}"):
        res = process_entry(entry)
        results.append(res)
        
        # Salvataggio incrementale (sovrascrive il file ad ogni iterazione per sicurezza)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)

    print(f"Saved results to: {OUTPUT_FILE}")

    # Fase 2: Calcolo Metriche immediato
    analyze_performance(results)

if __name__ == "__main__":
    main()