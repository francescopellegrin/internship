import os
import json
import pandas as pd
import boto3
import spacy
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
ROOT_DIR = os.path.dirname(GRANDPARENT_DIR)

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

# Assicurati di mettere i percorsi reali qui
INPUT_FILE = os.path.join(GRANDPARENT_DIR, "inference/rag/results/gemma-4b.json") 
OUTPUT_FILE = os.path.join(CURRENT_DIR, "results/gemma-4b.xlsx")

# AWS Bedrock Config
JUDGE_INFERENCE_PROFILE = os.getenv("INFERENCE_PROFILE_CLAUDE_OPUS_4_5")
REGION_NAME = os.getenv("AWS_REGION", "eu-west-1")

if not JUDGE_INFERENCE_PROFILE:
    print("WARNING: INFERENCE_PROFILE not found in .env.")

JUDGE_CONFIG = {
    "maxTokens": 1024,
    "temperature": 0.0 
}

# =================================================

class ThesisEvaluator:
    def __init__(self):
        print("Initializing Evaluator...")
        self.bedrock = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)
        print("Loading Spacy (it_core_news_sm)...")
        try:
            self.nlp = spacy.load("it_core_news_sm")
        except OSError:
            print("Spacy model not found. Please run: python -m spacy download it_core_news_sm")
            exit(1)

    def calculate_entity_recall(self, gold_text, gen_text):
        """
        Static Metric: Extracts Nouns, Proper Nouns, and Numbers from Gold.
        Checks what % of them exist in Generated.
        """
        if not gold_text or not gen_text:
            return 0.0

        doc_gold = self.nlp(gold_text.lower())
        doc_gen = self.nlp(gen_text.lower())
        
        gold_keywords = set([
            token.text for token in doc_gold 
            if token.pos_ in ['NOUN', 'PROPN', 'NUM'] and len(token.text) > 2
        ])
        
        if not gold_keywords:
            return 0.0
        
        gen_tokens = set([token.text for token in doc_gen])
        found = gold_keywords.intersection(gen_tokens)
        
        recall = len(found) / len(gold_keywords)
        return recall * 10  # Scale 0-10        

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

    def evaluate_single_answer(self, question, gold, generated):
        prompt = f"""
        You are an impartial expert judge evaluating an AI assistant in the Energy Trading and Energy Market Regulation domain.
        Your goal is to assess the quality of a generated legal answer compared to a Gold Standard and the User Question.

        QUESTION: {question}
        GOLD STANDARD ANSWER: {gold}
        GENERATED ANSWER: {generated}

        Evaluate the Generated Answer on the following 6 specific criteria using a scale from 1 to 10.

        SCORING GUIDELINES (ANCHORS):
        - Range: 1 to 10. Use decimal values if needed (e.g., 7.5).
        - IMPORTANT CALIBRATION: Treat score 6 as SUFFICIENT / PASSING.
        - 1 = COMPLETELY WRONG / UNACCEPTABLE. Total failure.
        - 2 = MINIMAL / NON-ANSWER. Admits inability to answer when it should know.
        - 3-4 = WRONG WITH MAJOR ISSUES. Core legal framework is wrong.
        - 5 = BORDERLINE. Close to sufficient but with minor mistakes.
        - 6 = SUFFICIENT (PASS). Provides basic necessary knowledge.
        - 8 = GOOD. No major issues; only minor fixes needed.
        - 10 = EXCELLENT. Authoritative and precise.

        CRITERIA (Score each 1-10):
        1. ACCURACY: Factually correct figures, dates, articles.
        2. COMPLETENESS: Covers key clauses/exceptions. Penalize heavily if the answer is truncated due to token limits.
        3. ANSWER RELEVANCE: Directly answers the question.
        4. STYLE ADHERENCE: Professional Italian legal style. Cut-off sentences or missing conclusions are stylistic failures.
        5. CITATION QUALITY: Explicit and correct citations.
        6. CONTEXTUAL GROUNDING (EDISON): Application to Edison context.

        REQUIRED OUTPUT:
        Return ONLY a valid JSON object. Do not use Markdown formatting.
        You MUST include a "reasoning" field: a brief text (max 20 words) in Italian.

        JSON Structure:
        {{
        "accuracy": 0.0,
        "completeness": 0.0,
        "answer_relevance": 0.0,
        "style_adherence": 0.0,
        "citation_quality": 0.0,
        "contextual_grounding": 0.0,
        "reasoning": "string"
        }}
        """
        response_text = self.call_bedrock_judge(prompt)
        try:
            # Robust JSON extraction
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start == -1 or end == 0: raise ValueError("No JSON found")
            return json.loads(response_text[start:end])
        except:
            return {
                "accuracy": 1, "completeness": 1, "answer_relevance": 1, 
                "style_adherence" : 1, "citation_quality" : 1, "contextual_grounding": 1, 
                "reasoning": "Error parsing JSON"
            }

    def evaluate_robustness(self, question, answers_list):
        # Format the list clearly for the model
        generated_responses = "\n".join([f"Response {i+1}: {ans}" for i, ans in enumerate(answers_list)])
        
        prompt = f"""
        You are an expert evaluator assessing the stability and robustness of an AI legal assistant.
        Your task is to analyze 5 different responses generated by the same model for the same user question and determine if they are semantically consistent.

        QUESTION: {question}
        GENERATED RESPONSES: {generated_responses}

        Evaluate the "Robustness" on a scale from 1 to 10 based on the consistency of the legal advice.

        SCORING GUIDELINES:
        - 10 = Perfect Consistency.
        - 8 = High Consistency.
        - 5 = Moderate Inconsistency.
        - 3 = Significant Inconsistency.
        - 1 = Contradiction / Hallucination.

        REQUIRED OUTPUT:
        Return ONLY a valid JSON object.
        You MUST include a "reasoning" field: a brief text (max 15 words) in Italian.

        JSON Structure:
        {{
        "robustness_score": 0.0,
        "reasoning": "string"
        }}
        """
        
        response_text = self.call_bedrock_judge(prompt)
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start == -1 or end == 0: raise ValueError("No JSON found")
            data = json.loads(response_text[start:end])
            # Return tuple for unpacking
            return data.get('robustness_score', 1), data.get('reasoning', 'No reasoning')
        except:
            return 1.0, "Error parsing JSON"
        
    def process_item(self, item):
        q_text = item['original_question']
        gold = item['gold_answer']
        gen_list = item['generated_responses']

        # --- 1. STATIC METRICS (Entity Recall) ---
        recall_scores = [self.calculate_entity_recall(gold, ans) for ans in gen_list]
        static_score = sum(recall_scores) / len(recall_scores)

        # --- 2. JUDGE METRICS ---
        acc_list, comp_list, rel_list, style_list, cit_list, ground_list = [], [], [], [], [], []
        reasoning_list = []
        
        for i, ans in enumerate(gen_list):
            scores = self.evaluate_single_answer(q_text, gold, ans)
            acc_list.append(scores.get('accuracy', 1))
            comp_list.append(scores.get('completeness', 1))
            rel_list.append(scores.get('answer_relevance', 1))
            style_list.append(scores.get('style_adherence', 1))
            cit_list.append(scores.get('citation_quality', 1))
            ground_list.append(scores.get('contextual_grounding', 1))
            r_text = scores.get('reasoning', 'N/A')
            reasoning_list.append(f"[Ans {i+1}]: {r_text}")
        
        # Averages
        accuracy = sum(acc_list) / len(acc_list)
        completeness = sum(comp_list) / len(comp_list)
        relevance = sum(rel_list) / len(rel_list)
        style = sum(style_list) / len(style_list)
        citation = sum(cit_list) / len(cit_list)
        grounding = sum(ground_list) / len(ground_list)
        robusteness, rob_reasoning = self.evaluate_robustness(q_text, gen_list)
        
        judge_avg = (accuracy + completeness + relevance + style + citation + robusteness + grounding) / 7

        full_reasoning = " | ".join(reasoning_list) + " | [ROB]: " + rob_reasoning

        # --- 3. FINAL WEIGHTED SCORE ---
        # 80% Judge, 20% Static
        final_score = (judge_avg * 0.8) + (static_score * 0.2)

        return {
            "ID": item.get('id', 'N/A'),
            "Q&A score": item.get('score'),
            "Question": q_text,
            "Accuracy": accuracy,
            "Completeness": completeness,
            "Relevance": relevance,
            "Style": style,
            "Citation": citation,
            "Grounding": grounding,
            "Robustness": robusteness,
            "Entity Recall": static_score,
            "Judge Avg": judge_avg,
            "FINAL SCORE": final_score,
            "Reasoning": full_reasoning
        }

    def run_evaluation(self):
        if not os.path.exists(INPUT_FILE):
            print(f"File not found: {INPUT_FILE}")
            return

        data = []
        # Supporto sia per file JSON puro che JSONL
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                # Prova a leggere come lista JSON intera
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    data = json.load(f)
                else:
                    # Altrimenti leggi come JSONL
                    for line in f:
                        if line.strip(): data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        print(f"Starting evaluation on {len(data)} items...")
        results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.process_item, item) for item in data]
            
            for future in tqdm(as_completed(futures), total=len(data)):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Error processing item: {e}")

        df = pd.DataFrame(results)
        
        # --- CALCOLO MEDIE E AGGIUNTA RIGA FINALE ---
        if not df.empty:
            # Calcola le medie solo per le colonne numeriche
            numeric_cols = df.select_dtypes(include=['number']).columns
            means = df[numeric_cols].mean()
            
            # Crea la riga delle medie
            mean_row = {col: means[col] for col in numeric_cols}
            mean_row['ID'] = 'MEDIA TOTALE'
            mean_row['Question'] = '---'
            mean_row['Reasoning'] = '---'
            
            # Converti in DataFrame e concatena
            mean_df = pd.DataFrame([mean_row])
            df = pd.concat([df, mean_df], ignore_index=True)
            
            print("\n=== GLOBAL AVERAGES (Calculated) ===")
            print(means)
        
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluator = ThesisEvaluator()
    evaluator.run_evaluation()