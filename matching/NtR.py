import json
import os
import openai
import argparse
import time

def save_to_jsonl(output_path, topic_turn_id, covered, uncovered, recall, full_response_temp=None):
    result = {
        "topic_turn_id": topic_turn_id,
        "covered": covered,
        "uncovered": uncovered,
        "recall": recall,
        "response": full_response_temp
    }
    with open(output_path, "a") as file:
        json.dump(result, file)
        file.write("\n")

def claim_coverage_prompt(gold_claims, predicted_answer, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)

    covered_claims = []
    uncovered_claims = []

    for claim_id, claim_text in gold_claims.items():
        prompt_string = f"""
# Instruction: I will provide you with a response and a gold information piece. Your task is to determine whether the response captures this piece of information.
# Gold Information:
{claim_text}
# Response: 
{predicted_answer}
# Please answer the following:
Does the Response capture the Gold Information? Only respond with 'yes' or 'no' without further explanation.
# Answer (yes/no):
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_string}
            ],
            temperature=0.0,
            timeout=60
        )
        response_text = response.choices[0].message.content.strip().lower()
        
        if response_text == 'yes':
            covered_claims.append(claim_id)
        else:
            uncovered_claims.append(claim_id)
    
    recall = len(covered_claims) / (len(covered_claims) + len(uncovered_claims)) if (covered_claims or uncovered_claims) else 0
    return covered_claims, uncovered_claims, recall

def process_claim_coverage(input_path_nuggets, input_path_answer, output_dir, openai_api_key):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path_nuggets, 'r') as file:
        gold_claims_dict = json.load(file)
    
    with open(input_path_answer, 'r') as file:
        raw_answer_dict = json.load(file)
    
    model_recall = {}
    
    model_responses = {}
    for topic_turn, models in raw_answer_dict.items():
        for model_name, predicted_answer in models.items():
            model_responses.setdefault(model_name, {})[topic_turn] = predicted_answer
    
    for model_name, topic_turn_responses in model_responses.items():
        print(f"Processing model: {model_name}")
        output_file = os.path.join(output_dir, f"coverage_{model_name}.jsonl")
        recall_file = os.path.join(output_dir, f"recall_{model_name}.json")
        
        if os.path.exists(output_file):
            continue
        
        turn_recalls = []
        
        for topic_turn, predicted_answer in topic_turn_responses.items():
            # print(f"Processing topic-turn: {topic_turn}")
            if topic_turn not in gold_claims_dict:
                continue
            gold_claims = gold_claims_dict[topic_turn]
            if not gold_claims or not predicted_answer:
                continue
            
            covered, uncovered, recall = claim_coverage_prompt(gold_claims, predicted_answer, openai_api_key)
            turn_recalls.append(recall)
            save_to_jsonl(output_file, topic_turn, covered, uncovered, recall)
            # print(f"Recall: {recall:.2f}")
        
        model_recall[model_name] = sum(turn_recalls) / len(turn_recalls) if turn_recalls else 0
        
        with open(recall_file, "w") as f:
            json.dump({"model_recall": model_recall[model_name]}, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute coverage of claims based on model responses.")
    parser.add_argument("--gold_nuggets", type=str, required=True, help="Path to the gold nuggets JSON file.")
    parser.add_argument("--response_answer", type=str, required=True, help="Path to the predicted answers JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key.")
    
    args = parser.parse_args()
    process_claim_coverage(args.gold_nuggets, args.response_answer, args.output_dir, args.openai_api_key)
