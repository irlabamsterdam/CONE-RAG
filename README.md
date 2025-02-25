# Conversational Gold Nuggets

This repository contains the resource and framework of the resource paper **Conversational Gold: Evaluating Personalized Conversational Search System using Gold Nuggets**.

## Resources

All resources are available in the `resources/` directory, categorized as follows:  

- `resources/human_resources/` – Human extracted nuggets, human gold response and human matching study (with mturk interface).
- `resources/automatic_resources/` - LLM extracted nuggets, LLM-based passages judgement pool
- `resources/participant_resources/` - Participant submission response and their nuggetized version.


Further resource will be provided:

- **Topics & Qrels**: Topic and qrel files for TREC iKAT 2024 will be released on the [NIST website](https://trec.nist.gov).

## CONE-RAG

This section provides an example of how to use the **CONE-RAG** framework for Retrieval-Augmented Generation (RAG) evaluation. 

### Environment Setup

Install dependencies:  
```bash
conda env create -f environment.yml
``` 

```bash
mkdir outputs
```

### 💎 Nuggets Extraction

Run the following commands to extract nuggets:  

- **From responses:**  
  ```bash
  python -m extraction.nuggetizer \
    --api_key "your_actual_api_key" \
    --model_id "gpt-4o" \
    --output_pkl "outputs/response_nuggets.pkl" \
    --output_json "outputs/response_nuggets.json" \
    --input_answers "resources/participant_resources/automatic_responses.json" \
    --input_topics "2024_test_topics.json"
  ```  

- **From passages:**  
  use the above code but put the passage text rather than response in "responses.json" file.

### 🖇 Nuggets Matching

Execute the matching process with the following commands:  

- **Nugget-to-Nugget (NtN):**  
  ```bash
  python -m matching.NtN \
    --gold_nuggets "resources/human_resources/gold_nuggets_human.json" \
    --response_nuggets "resources/participant_resources/automatic_response_nuggets.json" \
    --output_dir "outputs/NtN"
  ```

- **Nugget-to-Response (NtR):**  
  ```bash
  python -m matching.NtR \
    --gold_nuggets "resources/human_resources/gold_nuggets_human.json" \
    --response_answer "resources/participant_resources/automatic_responses.json" \
    --output_dir "outputs/NtR" \
    --openai_api_key "your_actual_api_key"
  ```  

