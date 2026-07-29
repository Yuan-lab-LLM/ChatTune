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

import asyncio
import copy
import datetime
import json
import os

import gradio as gr
import httpx
import numpy as np
import requests
from diagnosis_treatment.prompt_template import (
    request_type_map,
    reversed_inpatient_fields,
    reversed_medical_fields,
    reversed_sub_medical_fields,
)
from fastapi import HTTPException
from pydub import AudioSegment

from ..config import args, inference_gradio_http_common_headers
from ..util import asr, inference_gradio_json_data, tts, write_to_file

path = os.getcwd()

async def distribute(msg, chatbot, json_display, json_file, module, json_md, branch, prompt_version, model_name):
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return *(await fetch_response(msg, json_display, json_file, module)), prompt_version, model_name

async def clientinfo(msg, chatbot, json_display, json_file, module, json_md, branch, prompt_version, model_name):
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return *(await fetch_response(msg, json_display, json_file, module)), prompt_version, model_name

async def basicmedicalrecord(msg, chatbot, json_display, json_file, module, json_md, branch, prompt_version, model_name):
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return *(await fetch_response(msg, json_display, json_file, module)), prompt_version, model_name

async def hospitalregister(msg, chatbot, json_display, json_file, module, json_md, branch, prompt_version, model_name):
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return *(await fetch_response(msg, json_display, json_file, module)), prompt_version, model_name

async def returnvisit(msg, chatbot, json_display, json_file, module, json_md, branch, prompt_version, model_name):
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return *(await fetch_response(msg, json_display, json_file, module)), prompt_version, model_name

async def hospitalguide(msg, chatbot, json_display, json_file, module, json_md, branch, prompt_version, model_name):
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return *(await fetch_response(msg, json_display, json_file, module, branch)), prompt_version, model_name

def chat_process(messages, json_diaplay_v, json_file, json_v:str, branch=None):
    if json_file == "":
        if branch != None:
            json_v = json_v + "_" + branch
        params = copy.deepcopy(inference_gradio_json_data[json_v])
        unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        file_name = f"{json_v}-{unique_id}.json"
    else:
        file_name = json_file
        params = json_diaplay_v
    hc = params['chat']['historical_conversations']
    #hc_bak = params['chat']['historical_conversations_bak']
    hc.append({'role':'user', 'content':str(messages)})

    return params, file_name

def send_to_tab(from_data, to_data, module):
    from_data = json.loads(from_data)
    to_data = json.loads(to_data)
    try:
        match module:#to
            case "diagnosis":
                to_data['input']['client_info'] = from_data['input']['client_info']
                to_data['input']['basic_medical_record'] = from_data['output']['basic_medical_record']
            case "examass" | "scheme":
                to_data['input'].update({'client_info': from_data['input']['client_info'],
                    'basic_medical_record': from_data['input']['basic_medical_record'],
                    'diagnosis': from_data['output']['diagnosis']})
            case "returnvisit":
                to_data = inference_gradio_json_data['returnvisit']
                to_data['input'].update({'client_info': from_data['input']['client_info'],
                    'basic_medical_record': from_data['input']['basic_medical_record'],
                    'diagnosis': from_data['output']['diagnosis']})
                to_data['output']['return_visit'].update({'summary': "", 'if_visit': ""})
                to_data['chat'].update({'historical_conversations_bak': [], 'historical_conversations': []})
            case "default" | "surgical" | "chemo" | "radiation" | "psycho" | "rehabilitation" | "physical" | "alternative" | "observation":
                to_data['output'][f'{module}_therapy']['method'] = from_data['output'][f'{module}_therapy']['method']
        from_data = json.dumps(from_data, ensure_ascii=False, indent=4)
        to_data = json.dumps(to_data, ensure_ascii=False, indent=4)
    except:
        raise HTTPException(status_code=400, detail="Please check the input and output.")
    return from_data, to_data

async def fetch_response(msg, json_display, json_file, module, branch=None):
    json_display = json.loads(json_display)
    if module == "hospitalguide":
        params, json_file = chat_process(msg, json_display, json_file, module, branch)
        url = f"http://{args.host}:{args.port}/inference?request_type={request_type_map[module]}&scheme={branch}"
    else:
        params, json_file = chat_process(msg, json_display, json_file, module)
        url = f"http://{args.host}:{args.port}/inference?request_type={request_type_map[module]}"

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=inference_gradio_http_common_headers, json=params, timeout=600) as response:
            response_json = None
            if response.status_code == 200:
                response_str = ""
                async for chunk in response.aiter_bytes(chunk_size=65536 * 2):
                    chunk=chunk.decode('utf-8')
                    response_str += chunk
                    print(chunk)
                if '{"input":' and "output" in chunk:
                    json_str = '{"input":' + chunk.split('{"input":')[1]
                    response_json = json.loads(json_str)
                else:
                    print(f"error response {response_str}")
            if response_json:
                print(f"\n请求结果:{response_json}")
                new_history = [v['content'] for v in response_json['chat']['historical_conversations']]
                if len(new_history) % 2:
                    new_history.insert(0, None)
                new_history = np.array(new_history).reshape(int(len(new_history)/2),2)
                new_history = new_history.tolist()
                json_display = response_json

                write_flag = write_to_file(json_file, json_display)
                json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
                if write_flag == True:
                    json_md=f"""Successfully! write data to {json_file}.
    \nSuccessfully! write data to {str(json_file).replace('json', 'xlsx')}."""
                else:
                    json_md=""

                return None, new_history, json_display, json_file, module, json_md, branch

            else:
                return "Error: Unable to fetch response from inference API."

def user_response_stream(json_display, prompt_version, model_name):
    # chatbot = chatbot + [[msg, None]]
    json_display = json.loads(json_display)
    json_display['chat']['prompt_version'] = prompt_version
    json_display['chat']['model_name'] = model_name
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return json_display

def fetch_response_stream(msg, json_display, json_file, module, branch=None):
    json_display = json.loads(json_display)
    if module == "hospitalguide":
        params, json_file = chat_process(msg, json_display, json_file, module, branch)
        url = 'http://' + str(args.host) + ':' + str(args.port) + '/inference?request_type=' + request_type_map[module] + "&scheme=" + str(branch)
    else:
        params, json_file = chat_process(msg, json_display, json_file, module)
        url = 'http://' + str(args.host) + ':' + str(args.port) + '/inference?request_type=' + request_type_map[module]
    response = requests.post(url, headers=inference_gradio_http_common_headers,stream=True,json=params)
    json_display_dict=params
    list_new={'role': 'assistant', 'content': ''}
    json_display_dict['chat']['historical_conversations'].append(list_new)

    if response.status_code == 200:
        answer=""
        for chunk in response.iter_content(chunk_size=65536*2):
            chunk=chunk.decode('utf-8')
            if "input" and "output" in chunk:
                out_json=chunk
                print(f"\n请求结果:{out_json}")
                out_dict = json.loads(out_json)
                new_history = [v['content'] for v in out_dict['chat']['historical_conversations']]
                if len(new_history) % 2:
                    new_history.insert(0, None)
                new_history = np.array(new_history).reshape(int(len(new_history)/2),2)
                new_history = new_history.tolist()
                json_display = out_dict
                write_flag = write_to_file(json_file, json_display)
                json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
                if write_flag == True:
                    json_md=f"""Successfully! write data to {json_file}.
    \nSuccessfully! write data to {str(json_file).replace('json', 'xlsx')}."""
                    if module == "basicmedicalrecord": 
                        yield None, new_history, json_display, json_file, json_md
                        return
                else:
                    json_md=""
                
            else:
                json_md=""
                answer+=chunk
                json_display_dict['chat']['historical_conversations'][-1]['content']=answer
                new_history = [v['content'] for v in json_display_dict['chat']['historical_conversations']]
                if len(new_history) % 2:
                    new_history.insert(0, None)
                new_history = np.array(new_history).reshape(int(len(new_history)/2),2)
                new_history = new_history.tolist()
                json_display = json_display_dict
                json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
                yield None, new_history, json_display, json_file, json_md
        
        out_dict = json.loads(out_json)
        new_history = [v['content'] for v in out_dict['chat']['historical_conversations']]
        if len(new_history) % 2:
            new_history.insert(0, None)
        new_history = np.array(new_history).reshape(int(len(new_history)/2),2)
        new_history = new_history.tolist()
        json_display = out_dict
        json_display = json.dumps(json_display, ensure_ascii=False, indent=4)

        yield None, new_history, json_display, json_file, json_md
        return
    else:
        yield None ,"Error: Unable to fetch response from inference API." , None, None, None, None
        return

async def fetch_response_nochat(json_display, json_file, module, json_md, result_text, result_json, branch=None):
    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    json_file = f"{module}-{unique_id}.json"
    json_display_dict = json.loads(json_display)
    if module in ["doctormedicalrecord", "inpatient"]:
        url = f"http://{args.host}:{args.port}/inference?request_type={request_type_map[module]}&scheme={branch}"
    else:
        url = f"http://{args.host}:{args.port}/inference?request_type={request_type_map[module]}"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=json_display_dict,
            timeout=240,
        )
        if response.status_code == 200:
            print(f"\n请求结果:{response.json()}")
            results_json = response.json()
            results = ""
            if module == "diagnosis":
                for v in response.json()['output']['diagnosis']:
                    results += f"【{v['diagnosis_name']}（{v['diagnosis_name_retrieve']}）    {v['diagnosis_code']}    {v['diagnosis_identifier']}】\n"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump((results_json), f, ensure_ascii=False, indent=4)
                f.close()
                json_md=f"""Successfully! write data to {json_file}."""
                results_json = json.dumps(results_json, ensure_ascii=False, indent=4)
                return json_display, json_file, module, json_md, results, results_json, branch, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

            if module == "examass":
                for v in response.json()['output']['examine_content']:
                    exam_cd = "、".join([j['diagnosis_name'] for j in v['corresponding_diagnosis']])
                    results += f"""【检查名称: {v['examine_name']}（{v['examine_name_retrieve']}）】
检查编号: {v['examine_code']}
检查类别: {v['examine_category']}
开单数量: {v['order_quantity']}
针对疾病: {exam_cd}\n\n\n"""
                for v in response.json()['output']['assay_content']:
                    assay_cd = "、".join([j['diagnosis_name'] for j in v['corresponding_diagnosis']])
                    results += f"""【化验名称: {v['assay_name']}（{v['assay_name_retrieve']}）】
化验编号: {v['assay_code']}
化验类别: {v['assay_category']}
开单数量: {v['order_quantity']}
针对疾病: {assay_cd}\n\n\n"""
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump((results_json), f, ensure_ascii=False, indent=4)
                f.close()
                json_md=f"""Successfully! write data to {json_file}."""
                results_json = json.dumps(results_json, ensure_ascii=False, indent=4)
                return json_display, json_file, module, json_md, results, results_json, branch

            if module == "doctormedicalrecord":
                basic_medical_record = response.json()['output']['basic_medical_record']
                for key, value in basic_medical_record.items():
                    if not isinstance(value, dict):
                        results += f"【{reversed_medical_fields[key]}】：{value}\n" if value != "" else ""
                    else:
                        if not all(v == "" for v in value.values()):
                            results += f"【{reversed_medical_fields[key]}】\n"
                            for k, v in value.items(): results += f"  {reversed_sub_medical_fields[k]}: {v}\n" if v != "" else ""
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump((results_json), f, ensure_ascii=False, indent=4)
                f.close()
                json_md=f"""Successfully! write data to {json_file}."""
                results_json = json.dumps(results_json, ensure_ascii=False, indent=4)
                return json_display, json_file, module, json_md, results, results_json, branch

            if module == "inpatient":
                inpatient = response.json()['output']['inpatient']
                for key, value in inpatient.items():
                    if isinstance(value, dict):
                        results += f"【{reversed_inpatient_fields.get(key)}】：\n"
                        for k, v in value.items():
                            results += f"{reversed_inpatient_fields.get(k)}: {v}\n"
                    elif isinstance(value, list):
                        results += f"【{reversed_inpatient_fields.get(key)}】：\n"
                        for v in value:
                            results += f"{v['diagnosis_name']}（{v['diagnosis_name_retrieve']}）    {v['diagnosis_code']}    {v['diagnosis_identifier']}\n"
                    else:
                        results += f"【{reversed_inpatient_fields.get(key)}】：{value}\n"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump((results_json), f, ensure_ascii=False, indent=4)
                f.close()
                json_md=f"""Successfully! write data to {json_file}."""
                results_json = json.dumps(results_json, ensure_ascii=False, indent=4)
                return json_display, json_file, module, json_md, results, results_json, branch
        else:
            return "Error: Unable to fetch response from inference API."

async def fetch_response_pick_scheme(json_display, json_file, module):
    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    json_file= f"{module}-{unique_id}.json"
    json_display_dict = json.loads(json_display)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://{args.host}:{args.port}/inference?request_type={request_type_map[module]}&scheme=pick_therapy",
            json=json_display_dict,
            timeout=240,
        )
        if response.status_code == 200:
            print(f"\n请求结果:{response.json()}")
            result_json = response.json()
            result_text = ""
            line_str = "-"*40
            for v in response.json()['output']['pick_therapy']:
                _result_text = ""
                for _v in v['therapy_interpret']:
                    _result_text += f"""【{_v['therapy_name']}】：\n{_v['therapy_content']}\n"""
                result_text += f"""{line_str}{v['therapy_name']}{line_str}
方案概述: {v['therapy_summary']}
{_result_text}\n\n\n"""
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump((result_json), f, ensure_ascii=False, indent=4)
            f.close()
            json_md=f"""Successfully! write data to {json_file}."""
            result_json = json.dumps(result_json, ensure_ascii=False, indent=4)
            therapy_num = len(response.json()['output']['pick_therapy'])
            return json_file, json_md, result_text, result_json, result_json, gr.update(choices=[str(i) for i in range(1, therapy_num+1)], value="1")
        else:
            return "Error: Unable to fetch response from inference API."

async def fetch_response_generate_therapy(json_display, json_file, module, json_md, result_text, result_json, branch=None):
    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    json_file = f"{module}-{unique_id}.json"
    json_display_dict = json.loads(json_display)

    url = f"http://{args.host}:{args.port}/inference?request_type=v6&scheme=generate_therapy&sub_scheme={branch}&enable_think=1"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=json_display_dict,
            timeout=360,
        )
        if response.status_code == 200:
            print(f"\n请求结果:{response.json()}")
            results_json = response.json()
            results = ""
            line_str = "-"*40
            for k, v in response.json()['output']['generate_therapy'][0]['therapy_content'].items():
                if k == "prescription" and v != []:
                    results += f"{line_str}处方：{line_str}\n"
                    for _v in v:
                        results += f"""【药品名称: {_v['drug_name']}  （{_v['drug_name_retrieve']}）】
适应疾病: {_v['corresponding_diseases']}
药品作用: {_v['drug_efficacy']}
药品信息: 
    药品编号: {[_v['drug_id']]}
    药品规格: {_v['drug_specification']}
    用药途径: {_v['route_of_administration']}
开药信息:
    开单数量: {_v['order_quantity']}
    开单单位: {_v['order_unit']}
    单次剂量: {_v['dosage']}
    用药频次: {_v['frequency']}
    持续时间: {_v['duration']}\n\n\n"""

                if k == "transfusion" and v != []:
                    results += f"{line_str}输液：{line_str}\n"
                    for _v in v:
                        results += f"""【药品名称: {_v['drug_name']}  （{_v['drug_name_retrieve']}）】
适应疾病: {_v['corresponding_diseases']}
药品作用: {_v['drug_efficacy']}
药品信息: 
    药品编号: {[_v['drug_id']]}
    药品规格: {_v['drug_specification']}
    用药途径: {_v['route_of_administration']}
开药信息:
    开单数量: {_v['order_quantity']}
    开单单位: {_v['order_unit']}
    单次剂量: {_v['dosage']}
    用药频次: {_v['frequency']}
    持续时间: {_v['duration']}
    输液分组: {_v['infusion_group']}
    输液速度: {_v['infusion_rate']}\n\n\n"""

                if k == "disposition" and v != []:
                    results += f"{line_str}处置：{line_str}\n"
                    for _v in v:
                        results += f"""【处置名称: {_v['disposition_name']}】
处置编号: {_v['disposition_id']}
单次用量: {_v['dosage']}
处置频次: {_v['frequency']}
持续时间: {_v['duration']}\n\n\n"""

                if k == "examine" and v != []:
                    results += f"{line_str}检查：{line_str}\n"
                    for _v in v:
                        exam_cd = "、".join([j['diagnosis_name'] for j in _v['corresponding_diagnosis']])
                        results += f"""【检查名称: {_v['examine_name']}（{_v['examine_name_retrieve']}）】
检查编号: {_v['examine_code']}
检查类别: {_v['examine_category']}
开单数量: {_v['order_quantity']}
针对疾病: {exam_cd}\n\n\n"""

                if k == "assay" and v != []:
                    results += f"{line_str}化验：{line_str}\n"
                    for _v in v:
                        assay_cd = "、".join([j['diagnosis_name'] for j in _v['corresponding_diagnosis']])
                        results += f"""【化验名称: {_v['assay_name']}（{_v['assay_name_retrieve']}）】
化验编号: {_v['assay_code']}
化验类别: {_v['assay_category']}
开单数量: {_v['order_quantity']}
针对疾病: {assay_cd}\n\n\n"""
                if k in ["surgical", "chemo", "radiation", "psycho", "rehabilitation", "physical", "alternative", "observation"] and v != []:
                    results += f"{line_str}治疗方法：{line_str}\n"
                    for _v in v:
                        results += f"""【治疗名称: {_v['method_name']}】
治疗编号: {_v['method_code']}
治疗类型: {_v['method_type']}
适用疾病: {_v['corresponding_diseases']}
治疗计划: {_v['method_plan']}
潜在风险: {_v['method_risk']}\n\n\n"""

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump((results_json), f, ensure_ascii=False, indent=4)
            f.close()
            json_md=f"""Successfully! write data to {json_file}."""
            results_json = json.dumps(results_json, ensure_ascii=False, indent=4)
            return json_display, json_file, module, json_md, results, results_json, branch
        else:
            return "Error: Unable to fetch response from inference API."

async def fetch_response_generate_all_therapy(json_display, json_file, module, json_md, result_text, result_json, branch=None):
    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    json_file = f"{module}-{unique_id}.json"
    json_display_dict = json.loads(json_display)

    url = f"http://{args.host}:{args.port}/inference?request_type=v6&scheme=pick_therapy"
    async with httpx.AsyncClient(timeout=240) as client:
        response = await client.post(url, json=json_display_dict)
        output = copy.deepcopy(response.json())
        print(f"\n请求结果:{response.json()}")

    therapy_num = len(output['output']['pick_therapy'])
    sema = asyncio.Semaphore(6)
    async def generate_therapy(index: int):
        nonlocal output
        url = f"http://{args.host}:{args.port}/inference?request_type=v6&scheme=generate_therapy&sub_scheme={index}&enable_think=1"
        async with sema:
            async with httpx.AsyncClient(timeout=240) as client:
                response = await client.post(url, json=output)
                output['output']['generate_therapy'].append(response.json()['output']['generate_therapy'][0])
                print(f"\n请求结果:{response.json()}")

    tasks = [asyncio.create_task(generate_therapy(index)) for index in range(1, therapy_num+1, 1)]
    await asyncio.gather(*tasks)

    output['output']['generate_therapy'].pop(0)
    result_json = json.dumps(output, ensure_ascii=False, indent=4)
    return json_display, json_file, module, json_md, result_text, result_json, branch

SAVE_DIR = "./audios"
os.makedirs(SAVE_DIR, exist_ok=True)
def save_audio(audio):
    if not audio:
        return "", "Not Found: Please Record First."

    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    type = audio.split(".")[-1]
    save_name = f"record_{unique_id}.{type}"
    save_path = os.path.join(SAVE_DIR, save_name)
    if type == "mp3":
        audio = AudioSegment.from_mp3(audio)
    elif type == "wav":
        audio = AudioSegment.from_wav(audio)
    else:
        return "", "Error: Unsupported Media Type."

    audio.export(save_path, format=type)

    return save_path, f"The audio has been saved:  {save_path}"

def update_by_asr(save_path, json_display, asr_model):
    completion_text = asr(save_path, asr_model)
    json_display = json.loads(json_display)
    json_display['input']['doctor_supplement'] = completion_text
    json_display = json.dumps(json_display, ensure_ascii=False, indent=4)
    return "", json_display

def update_by_tts(enable_voice, chatbot, tts_model):
    if enable_voice == "YES":
        save_path = tts(SAVE_DIR, chatbot[-1][1], tts_model)
    else:
        save_path = None
    return save_path