import json
import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compute_entailment(premise, hypothesis, model, tokenizer, device):
    input_data = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(input_data["input_ids"])
    prediction_probs = torch.softmax(output.logits[0], -1).tolist()
    return {
        "entailment": round(float(prediction_probs[0]) * 100, 1),
        "neutral": round(float(prediction_probs[1]) * 100, 1),
        "contradiction": round(float(prediction_probs[2]) * 100, 1),
    }

def process_nuggets(response_nuggets_path, gold_nuggets_path, output_dir, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    response_nuggets = load_json(response_nuggets_path)
    gold_nuggets = load_json(gold_nuggets_path)
    
    entailment_results = {}
    predicted_gold_nuggets = {}
    predictive_response_nuggets = {}
    all_nuggets = {}
    
    for model_name in set(model for turn in response_nuggets.values() for model in turn):
        entailment_results[model_name] = {}
        predicted_gold_nuggets[model_name] = {}
        predictive_response_nuggets[model_name] = {}
        all_nuggets[model_name] = {}

        # Compute entailment scores
        for turn_id, model_data in tqdm(response_nuggets.items()):
            if model_name not in model_data:
                print(f"Model {model_name} not found in turn {turn_id}", flush=True)
                continue
            predicted_gold_nuggets[model_name][turn_id] = set()
            predictive_response_nuggets[model_name][turn_id] = set()
            all_nuggets[model_name][turn_id] = set()
            
            data = model_data[model_name]
            if "nuggets" not in data:
                print(f"No nuggets found for model {model_name} in turn {turn_id}", flush=True)
                continue
            
            response_nuggets_list = data["nuggets"].get("matched", [])
            response_nugget_ids = {f"[{i+1}]": nugget for i, nugget in enumerate(response_nuggets_list)}
            all_nuggets[model_name][turn_id].update(response_nugget_ids.keys())
            
            for gold_id, gold_data in gold_nuggets.get(turn_id, {}).items():
                gold_text = gold_data["text"]
                
                for resp_id, response_nugget in response_nugget_ids.items():
                    entailment_score = compute_entailment(response_nugget, gold_text, model, tokenizer, device)
                    if entailment_score["entailment"] > 50.0:
                        predictive_response_nuggets[model_name][turn_id].add(resp_id)
                        predicted_gold_nuggets[model_name][turn_id].add(gold_id)
        
        # Compute precision and recall
        turn_precisions = []
        turn_recalls = []
        for turn_id in all_nuggets[model_name]:
            all_nugget_count = len(all_nuggets[model_name][turn_id])
            all_gold_nugget_count = len(gold_nuggets[turn_id])
            if all_nugget_count != 0:
                precision = len(predictive_response_nuggets[model_name][turn_id]) / all_nugget_count if all_nugget_count > 0 else 0
                turn_precisions.append(precision)
            recall = len(predicted_gold_nuggets[model_name][turn_id]) / all_gold_nugget_count if all_gold_nugget_count > 0 else 0
            turn_recalls.append(recall)

        
        entailment_results[model_name]["precision"] = sum(turn_precisions) / len(turn_precisions) if turn_precisions else 0
        entailment_results[model_name]["recall"] = sum(turn_recalls) / len(turn_recalls) if turn_recalls else 0

        # break
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "entailment_results.json"), "w") as f:
        json.dump(entailment_results, f, indent=4)
    
    nuggets_output = {
        "predicted_gold_nuggets": {k: {tk: list(tv) for tk, tv in v.items()} for k, v in predicted_gold_nuggets.items()},
        "predictive_response_nuggets": {k: {tk: list(tv) for tk, tv in v.items()} for k, v in predictive_response_nuggets.items()}
    }
    
    with open(os.path.join(output_dir, "nuggets_results.json"), "w") as f:
        json.dump(nuggets_output, f, indent=4)
    
    print("Entailment computation completed and results saved in", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute entailment between response and gold nuggets.")
    parser.add_argument("--gold_nuggets", type=str, required=True, help="Path to the gold nuggets JSON file.")
    parser.add_argument("--response_nuggets", type=str, required=True, help="Path to the response nuggets JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--model_name", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", help="Model name for entailment.")
    
    args = parser.parse_args()
    process_nuggets(args.response_nuggets, args.gold_nuggets, args.output_dir, args.model_name)
