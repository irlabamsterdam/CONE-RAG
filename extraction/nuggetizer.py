
# %%
import json
from openai import OpenAI
import pickle
import os
import re
import argparse

from extraction.functions import load_ikat_dataset, call_gpt, extract_nuggets, check_nugget_exists, run_nugget_again, check_nugget_exist_two


prompt_response_extraction = """I will give you a user query and a response to the user query. You should extract the nuggets of information related to the user query from the given response. The nuggets should be an exact copy of a span of text from the response. 
Please extract the nuggets and write each nugget in one line. If there is no nugget of information in the given text, please only say "No nugget".
## user query: {query}
## document: {doc}
## nuggets: 
# (Please copy exact spans from the text as nuggets)"""

def extract_nuggets_from_response(response, user_query, model_id, client):

    nuggets_arr = {'matched': [],
                   'not-matched': []}   
    prompt_input = prompt_response_extraction.format(doc=response, query=user_query)
    text, conversations = call_gpt(prompt_input, client, model_id)
    extracted_nuggets = extract_nuggets(text)
    
    for text_nug in extracted_nuggets:
        print('*'*5)
        print(text_nug)
        
        if "No nugget" in text_nug:
            continue

        if check_nugget_exists(text_nug, response):
            print('yes')
            nuggets_arr['matched'].append(text_nug)

        else:
            print('no')
            new_nugget_text = run_nugget_again(conversations, text_nug, client, model_id)
            print(new_nugget_text)
            bool_var, new_nuggets_arr = check_nugget_exist_two(new_nugget_text, response)
            
            if bool_var:
                print('the nugget matched in the second call.')
                nuggets_arr['matched']+= new_nuggets_arr
            
            else:
                print("couldn't match the nugget in the second call")
                nuggets_arr['not-matched'].append(new_nugget_text)

    print('*'*20)
    
    nuggets_arr['matched'] = list(set(nuggets_arr['matched']))
    
    return nuggets_arr

def clean_response(response):
    response = response.replace('*', ' ')
    response = re.sub(r'\s+', ' ', response)
    return response

def run_nuggetizer(client, model_id, output_path_to_extracted_nuggets_pkl, output_path_to_extracted_nuggets_json,
                   input_path_to_answers, input_topics_path):

    with open(input_path_to_answers, 'r') as f:
        data_responses = json.load(f)

    user_utterance = load_ikat_dataset(input_topics_path)

    team_nuggets = {}

    if os.path.exists(output_path_to_extracted_nuggets_pkl):
        with open(output_path_to_extracted_nuggets_pkl, 'rb') as f:
            team_nuggets = pickle.load(f)
        
    index = 0

    for turn_id in data_responses:
        index += 1
        print(index)
        user_query = user_utterance[turn_id]
        
        if not (turn_id in team_nuggets):
            team_nuggets[turn_id] = {}
        
        print(f'User query: {user_query}')

        for team_name in data_responses[turn_id]:
            print(f'Team name: {team_name}')
            
            if not (team_name in team_nuggets[turn_id]):
                response = data_responses[turn_id][team_name]
                response = clean_response(response)
                print(f'Team response: {response}')
                nuggets = extract_nuggets_from_response(response, user_query, model_id, client)
                team_nuggets[turn_id][team_name] = {'nuggets':nuggets,
                                                'response': response}

        with open(output_path_to_extracted_nuggets_pkl, 'wb') as f:
            pickle.dump(team_nuggets, f)
        print('Pickle saved')

    with open(output_path_to_extracted_nuggets_pkl, 'wb') as f:
        pickle.dump(team_nuggets, f)
    print('Pickle saved')

    with open(output_path_to_extracted_nuggets_json, 'w') as fout:
        json_dumps_str = json.dumps(team_nuggets, indent=4)
        print(json_dumps_str, file=fout)

    return

def main(api_key, model_id, output_path_to_extracted_nuggets_pkl, output_path_to_extracted_nuggets_json,
         input_path_to_answers, input_topics_path):


    client = OpenAI(api_key=api_key)

    run_nuggetizer(client, model_id, output_path_to_extracted_nuggets_pkl, output_path_to_extracted_nuggets_json,
                   input_path_to_answers, input_topics_path)

    print(f'Nugget extraction completed. Find the results here: {output_path_to_extracted_nuggets_json}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the nugget extraction process with custom arguments."
    )
    
    # API key and model id
    parser.add_argument('--api_key', type=str, default="",
                        help="Your API key for authentication.")
    parser.add_argument('--model_id', type=str, default="gpt-4o",
                        help="The model identifier.")
    # Output file paths
    parser.add_argument('--output_pkl', type=str, default='outputs/response_nuggets.pkl',
                        help="Path to the output pickle file for extracted nuggets.")
    parser.add_argument('--output_json', type=str, default="outputs/response_nuggets.json",
                        help="Path to the output JSON file for extracted nuggets.")
    # Input file paths
    parser.add_argument('--input_answers', type=str, default="resources/participant_resources/automatic_responses.json",
                        help="Path to the JSON file containing participant responses.")
    parser.add_argument('--input_topics', type=str, default="2024_test_topics.json",
                        help="Path to the JSON file containing test topics.")
    
    args = parser.parse_args()
    
    main(api_key=args.api_key, 
         model_id=args.model_id, 
         output_path_to_extracted_nuggets_pkl=args.output_pkl, 
         output_path_to_extracted_nuggets_json=args.output_json, 
         input_path_to_answers=args.input_answers, 
         input_topics_path=args.input_topics)
    