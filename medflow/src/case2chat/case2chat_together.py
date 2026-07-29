# Copyright (c) 2025,  IEIT SYSTEMS Co.,Ltd.  All rights reserved

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import warnings
import datetime
from typing import List
from typing_extensions import Annotated

import argparse
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Body
from fastapi.encoders import jsonable_encoder
from openai import OpenAI

from prompt_together import PromptTemplate
from multiprocessing import  freeze_support

def args_parser():
    """Argument parser setup"""
    parser = argparse.ArgumentParser(description='Chatbot Interface with Customizable Parameters')
    parser.add_argument('--model-url', type=str, default='http://localhost:8000/v1', help='Model URL')
    parser.add_argument('--model', type=str, required=True, help='Model name for the chatbot')
    parser.add_argument('--log', action='store_true', help='If True, save log to ./medical_xxx.log.')
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=8016)
    args = parser.parse_args()
    return args

app = FastAPI()

class BasicMedicalRecord(BaseModel):
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: str
    personal_history: str
    allergy_history: str
    physical_examination: str
    auxiliary_examination: str
    
class HistoricalConversations(BaseModel):
    role: str
    content: str
    
def preprocess(params):
    """Prompt preprocess"""
    prompt = PromptTemplate(params['basic_medical_record'])
    system_str, user_str = prompt.generate_prompt()
    messages = [
        {"role": "system", "content": system_str},
        {"role": "user", "content": user_str}
    ]
    # print(f"初始prompt: \n{messages=}\n")
    return messages

async def predict(messages):
    """Inference"""
    args = args_parser()
    client = OpenAI(api_key="EMPTY", base_url=args.model_url)
    chat_response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=1024,#8192
        stream=False,
        stop="<|eot_id|>",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    answer=chat_response.choices[0].message.content
    # print(f"对话结果: {answer=}")
    return answer

def extract_json_data(answer):
    """Check results, extract conversation."""
    chat_format = r'\[(.*?)\]'
    chat_data = answer.replace("\n", "").replace("\t", "").replace(" ", "")
    chat_search = re.search(chat_format, chat_data, re.DOTALL)

    if isinstance(chat_search, re.Match):
        chat_search = chat_search.group(0)
        chat_search = eval(chat_search)
    # print(f"\n大模型匹配内容：\n{chat_search=}")
 
    return chat_search
    
def postprocess(receive, answer):
    """Result postprocess"""
    params = receive.copy()
    chat_search = extract_json_data(answer)
    
    history_conversations = []
    for item in chat_search:
        history_conversations.append(HistoricalConversations(
            role = item['role'],
            content = item['content']
        ))
    params['historical_conversations'] = history_conversations

    return params

def save_log(messages, answer, results):
    """Save results to file"""
    global curl_num
    with open(f'medical_{log_name}.log', 'a') as f:
        f.writelines(f"\n=======================请求: {curl_num}===============================\n")
        f.writelines(f"\n1.初始prompt: \n{messages}\n")
        f.writelines(f"\n2.对话结果: {answer}\n")
        f.writelines(f"\n3.请求返回结果: \n{jsonable_encoder(results)}\n")
        f.writelines(f"\n=============================================================\n")
    f.close
    curl_num+=1

@app.get("/train/{item_id}")
async def update_item(
    item_id: int,
    item_type: str,
    basic_medical_record: BasicMedicalRecord,
    historical_conversations: List[HistoricalConversations],
    remark: Annotated[str, Body()]
):
    args = args_parser()
    receive = {
        "item_id": item_id,
        "item_type": item_type,
        "basic_medical_record": basic_medical_record,
        "historical_conversations": historical_conversations,
        "remark": remark
    }
    print(f"请求传入参数: \n{receive=}\n")
    
    match item_type:
        case "case2chat":
            messages = preprocess(receive)
            answer = await predict(messages)
            results = postprocess(receive, answer)
            print(f"请求返回结果：\n{results=}\n")
            
            # if args.log:
            #     save_log(messages, answer, results)
            
            return results
        case _:
            return "Error: item_type is incorrect, please change it."

if __name__ == '__main__':
    freeze_support()
    curl_num = 0
    log_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    args = args_parser()
    
    uvicorn.run(app='case2chat_together:app', host=args.host, port=args.port,workers=32,reload=False)
