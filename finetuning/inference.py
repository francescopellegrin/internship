import json
import os
import sys
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ================= CONFIGURAZIONE =================
WORK_DIR = os.environ.get("WORK")
MODEL_NAME = "gemma-3-27b_0_ft" 

INPUT_FILE = os.path.join(WORK_DIR, "offline_datasets", "testA.jsonl")
OUTPUT_FILE = os.path.join(WORK_DIR, "inference_results", f"{MODEL_NAME}.jsonl")
BASE_MODEL_PATH = os.path.join(WORK_DIR, "offline_models", "gemma-3-27b")
ADAPTER_PATH = os.path.join(WORK_DIR, "ft_models", MODEL_NAME)

NUM_ANSWERS = 5
GEN_CONFIG = {
    "max_new_tokens": 1024,      
    "temperature": 0.5,
    "top_p": 0.9,
    "do_sample": True,
    "num_return_sequences": NUM_ANSWERS 
}
# ==================================================

def load_model_optimized():
    print(f"Loading Tokenizer from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    # CRITICO PER BATCH INFERENCE: Padding a sinistra
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading Base Model (4-bit + SDPA) from {BASE_MODEL_PATH}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )

    # Usa SDPA come richiesto
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config, 
        device_map="cuda", 
        attn_implementation="sdpa", 
        trust_remote_code=True
    )

    if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
        print(f"Loading LoRA Adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        print(f"CRITICAL: Adapter path {ADAPTER_PATH} not found.")
        sys.exit(1)

    model.eval()
    return tokenizer, model

def process_inference():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Dataset not found at {INPUT_FILE}")
        sys.exit(1)

    tokenizer, model = load_model_optimized()

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    print(f"Reading dataset...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]
        except Exception:
            print("Error reading JSON.")
            sys.exit(1)

    print(f"Starting inference on {len(data)} items...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps({"_metadata": {"model": MODEL_NAME}}) + "\n")
        
        for item in tqdm(data):
            # Garbage Collection per evitare OOM su sequenze lunghe
            torch.cuda.empty_cache()
            
            messages = item.get("messages", [])
            chat_format = [msg for msg in messages if msg['role'] in ['system', 'user']]
            
            user_content = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), "")
            gold_answer = next((m['content'] for m in reversed(messages) if m['role'] == 'assistant'), "")

            # Tokenization
            inputs = tokenizer.apply_chat_template(
                chat_format,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            input_len = inputs.shape[1]

            # Generation PARALLELA (5 risposte in 1 colpo)
            try:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = model.generate(
                            inputs,
                            **GEN_CONFIG,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=terminators,
                            use_cache=True
                        )
            except Exception as e:
                print(f"Gen Error: {e}")
                continue

            generated_responses = []
            # Il modello ritorna [batch_size * num_return_sequences, seq_len]
            # Ma qui inputs ha batch=1, quindi outputs ha shape [5, seq_len]
            for output in outputs:
                out_toks = output[input_len:]
                text = tokenizer.decode(out_toks, skip_special_tokens=True).strip()
                generated_responses.append(text)

            result = {
                "id": item.get('id'),
                "original_question": user_content,
                "gold_answer": gold_answer,
                "generated_responses": generated_responses,
                "model_profile": MODEL_NAME
            }
            
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Done. Processed items saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_inference()