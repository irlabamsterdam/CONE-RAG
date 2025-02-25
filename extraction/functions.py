import os
import json
import numpy as np
import matplotlib.pyplot as plt
import json
from openai import OpenAI
import re



def load_ikat_dataset(topics_path):
    turn_uterance = {}

    with open(topics_path, 'r') as f:
        topics = json.load(f)
    
    for topic in topics:
        conv_id = topic['number']
        for turn in topic['turns']:
            q_id = str(conv_id) + '_' + str(turn['turn_id'])
            turn_uterance[q_id] = turn['resolved_utterance']

    return turn_uterance

def load_nuggets(data_dir, topics_path):
    all_documents = []
    turn_uterance = load_ikat_dataset(topics_path)
    file_names = os.listdir(data_dir)
    nuggets_ds = []

    for file_name in file_names:
        with open(data_dir+file_name, 'r') as f:
            content = json.load(f)
            turn_id = file_name.split('.')[1].split('.')[0]
            nugget = {'turn_id': turn_id, 
                      'user_utterance': turn_uterance[turn_id], 
                      'manual_response':content['manual_response'].strip(), 
                      'all_docs':[],'nuggets':[]}

            for elem in content['clips']:
                nugget['nuggets'].append({'docid': elem['docid'], 'nugget-text':elem['text'].strip() })
                nugget['all_docs'].append(elem['docid'])
                
                if not elem['docid'] in all_documents:
                    all_documents.append(elem['docid'])
            
            if (len(nugget['nuggets'])>0) or len(content['manual_response'].strip())>0:
                nuggets_ds.append(nugget)

    return nuggets_ds, all_documents

def call_gpt(prompt, client, model_id):

    conversations = []
    conversations.append({'role': 'user', 'content': prompt})
    response = chatgpt_conversation(conversations, client, model_id)
    generated_text = response.strip()
    conversations.append({'role': 'assistant', 'content': generated_text})

    return generated_text, conversations

def chatgpt_conversation(conversation_Log, client, model_id):
  
  response = client.chat.completions.create(
      model = model_id,
      messages= conversation_Log,
      temperature= 0,
      top_p= 1,
      n=1, 
  )
  
  response= response.choices[0].message.content

  return response

def run_nugget_again(conversation_Log, nugget_text, client, model_id):
    
    prompt = """This nugget is not an exact copy from the text. Please copy the exact span of the following nugget from the text: 
    ## nugget: {nugget_text}""".format(nugget_text=nugget_text)
    
    conversation_Log.append({'role': 'user', 'content': prompt})
    response = chatgpt_conversation(conversation_Log, client, model_id)

    return response

def extract_nuggets(generated_text):
    final_arr = []
    nuggets_arr = generated_text.strip().split('\n')

    for i in range(0, len(nuggets_arr)):
        tmp_text = nuggets_arr[i].strip()
        for elem in ['-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', "\"", "'"]:
            if tmp_text.startswith(elem):
                tmp_text = tmp_text.lstrip(elem).strip()
        
        for elem in [".", "!", "?", "\"", "'"]:
            if tmp_text.endswith(elem):
                tmp_text = tmp_text.rstrip(elem).strip()

        if len(tmp_text)>0:
            final_arr.append(tmp_text)

    return final_arr

def check_nugget_exists(nugget_text, doc):
    
    if nugget_text in doc:
        return True
    
    if nugget_text.lower() in doc.lower():
        return True

    nugget_text_tmp = re.sub(r'\s+', ' ', nugget_text.lower())
    doc_tmp = re.sub(r'\s+', ' ', doc.lower())

    if nugget_text_tmp in doc_tmp:
        return True

    return False

def check_nugget_exist_two(nugget_text, doc):
    nuggets_arr = extract_nuggets(nugget_text)
    final_nuggets_arr = []
    checke_var = False

    for nugget_text in nuggets_arr:
        if (check_nugget_exists(nugget_text, doc)):
            checke_var = True
            final_nuggets_arr.append(nugget_text)
            

    return checke_var , final_nuggets_arr

