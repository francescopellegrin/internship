import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from trl import SFTTrainer, SFTConfig

# --- 1. CONFIGURAZIONE PATH E AMBIENTE ---
WORK_DIR = os.environ.get("WORK", "/leonardo_work/AIFAC_F01_260")
os.environ["HF_HOME"] = os.path.join(WORK_DIR, ".cache", "huggingface")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

# MODIFICARE QUI I NOMI MODELLO/OUTPUT
MODEL_NAME = "llama3-2-3b-instruct" 
OUTPUT_NAME = "llama-3b_ft"

MODEL_PATH = os.path.join(WORK_DIR, "offline_models", MODEL_NAME)
OUTPUT_DIR = os.path.join(WORK_DIR, "ft_models", OUTPUT_NAME)
TRAINSET_PATH = os.path.join(WORK_DIR, "offline_datasets", "train.jsonl")
VALSET_PATH = os.path.join(WORK_DIR, "offline_datasets", "validation.jsonl")

def main():
    print(f"--- STARTING TRAINING FOR: {MODEL_NAME} ---")

    # --- 2. DATASET ---
    train_set = load_dataset("json", data_files=TRAINSET_PATH, split="train")
    validation_set = load_dataset("json", data_files=VALSET_PATH, split="train")

    # --- 3. QUANTIZZAZIONE (4-bit) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 4. CARICAMENTO MODELLO ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
        dtype=torch.bfloat16
    )
    
    # Abilita Gradient Checkpointing (Vitale per i modelli grandi futuri)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    # --- 5. TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "right"
    
    # Fix per modelli senza pad token (Llama 3 / Qwen)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 6. LORA CONFIG (Da cambiare in base al modello) ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # --- 7. TRAINING ARGUMENTS (Da cambiare in base alla VRAM) ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        weight_decay=0.05,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        packing=False,
        report_to="none",
        gradient_checkpointing=True
    )

    training_args.max_seq_length = 2048

    # --- 8. TRAINER ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=validation_set,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # --- 9. VALUTAZIONE FINALE SUL BEST MODEL ---
    print("\n" + "="*50)
    print(f"VALUTAZIONE COMPLETA DEL BEST MODEL: {OUTPUT_DIR}")
    print("="*50)

    # Esegue la valutazione sul validation set usando il modello migliore appena ricaricato
    metrics = trainer.evaluate()

    # Stampa formattata di TUTTE le metriche disponibili
    for key, value in metrics.items():
        # Formatta i numeri float a 4 decimali per pulizia, lascia il resto com'è
        if isinstance(value, float):
            print(f"{key:<30}: {value:.4f}")
        else:
            print(f"{key:<30}: {value}")

    print("="*50 + "\n")
    print(f"--- TRAINING FINISHED: {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()