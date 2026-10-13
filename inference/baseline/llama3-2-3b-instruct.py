import boto3
import json
import os
from botocore.exceptions import ClientError
from tqdm import tqdm
from dotenv import load_dotenv
from botocore.config import Config
import concurrent.futures

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)

# Load .env from grandparent directory
dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

# Paths
INPUT_FILE = os.path.join(PARENT_DIR, "test.jsonl")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "results/llama3-2-3b-instruct.jsonl")

INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_LLAMA_3_2_3B")

if not INFERENCE_PROFILE:
    print(f"WARNING: {INFERENCE_PROFILE} not found in .env. Code might fail.")

# Region (Bedrock Inference Profiles are often region-specific)
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

# Robustness Settings
NUM_ANSWERS = 5
MAX_CONCURRENT_QUESTIONS = 10  # Elabora 10 domande alla volta (50 thread totali)
INFERENCE_CONFIG = {
    "maxTokens": 1024,
    "temperature": 0.5, 
    "topP": 0.9
}
# =================================================

def get_bedrock_client():
    """
    Configura il client con 'adaptive retry'.
    Se AWS risponde con 'ThrottlingException',
    questo client aspetterà automaticamente e riproverà.
    """
    retry_config = Config(
        region_name=REGION_NAME,
        retries={
            'max_attempts': 10,
            'mode': 'adaptive'
        },
        read_timeout=900,
        connect_timeout=10
    )
    print(f"Connecting to Bedrock in {REGION_NAME}...")
    client =  boto3.client(
        service_name='bedrock-runtime', 
        config=retry_config
    )
    return client

# Initialize one client globally for reuse
bedrock_client_global = get_bedrock_client()

def generate_direct_response(messages_history, system_prompt=None):
    # Clona i messaggi per evitare modifiche by-reference indesiderate
    current_messages = [msg.copy() for msg in messages_history]
    
    # Preparazione del System Prompt per l'API Converse
    # L'API si aspetta una lista di dizionari: [{'text': '...'}]
    bedrock_system_prompts = []
    if system_prompt:
        bedrock_system_prompts = [{"text": system_prompt}]

    try:
        # Usiamo il client GLOBALE
        # Passiamo i messaggi puliti e il system prompt nel parametro dedicato
        response = bedrock_client_global.converse(
            modelId=INFERENCE_PROFILE,
            messages=current_messages,
            system=bedrock_system_prompts, 
            inferenceConfig=INFERENCE_CONFIG
        )
        
        output_message = response['output']['message']
        stop_reason = response.get('stopReason')
        full_text = "".join([block['text'] for block in output_message['content'] if 'text' in block])
        cleaned_text = full_text.strip()

        if not cleaned_text:
            if stop_reason == "max_tokens": return "[ERROR_WHITESPACE_LOOP]"
            return "[EMPTY_GENERATION]"

        if stop_reason == "max_tokens":
            return cleaned_text + " [CUT_OFF]"

        return cleaned_text

    except Exception as e:
        return f"[ERROR_CALL_FAILED: {str(e)}]"

def process_single_question(item):
    """
    Helper function to process one question (item) fully.
    It runs the N generations for this question in parallel.
    """
    raw_messages = item.get("messages", [])
    
    input_messages = []
    sys_prompt = None
    gold_ans = None
    last_user_input = None

    for msg in raw_messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "system":
            sys_prompt = content
        elif role == "user":
            # 'converse' API expects content to be a list of dicts for the user
            input_messages.append({"role": "user", "content": [{"text": content}]})
            last_user_input = content
        elif role == "assistant":
            gold_ans = content 

    if not last_user_input:
        return None

    generated_answers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ANSWERS) as executor:
        # Submit the same task 5 times
        futures = [executor.submit(generate_direct_response, input_messages, sys_prompt) for _ in range(NUM_ANSWERS)]
        
        for future in concurrent.futures.as_completed(futures):
            generated_answers.append(future.result())

    # Structure Result
    return {
        "id": item.get("index"),
        "original_question": last_user_input,
        "gold_answer": gold_ans,
        "score": item.get("score"),
        "generated_responses": generated_answers,
        "model_profile": INFERENCE_PROFILE
    }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Dataset not found at {INPUT_FILE}")
        return
    
    print(f"Loading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            questions = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            questions = [json.loads(line) for line in f if line.strip()]

    print(f"Starting inference on {len(questions)} questions using profile: {INFERENCE_PROFILE}")
    print(f"Parallel mode: {MAX_CONCURRENT_QUESTIONS} questions concurrently x {NUM_ANSWERS} answers each.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_QUESTIONS) as executor:
            
            # Sottomettiamo tutte le domande
            futures = [executor.submit(process_single_question, item) for item in questions]
            
            # Usiamo tqdm su as_completed per vedere la barra di progresso avanzare man mano che finiscono
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(questions)):
                try:
                    result = future.result()
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                except Exception as e:
                    print(f"\nError processing a question: {e}")

    print(f"Done. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()