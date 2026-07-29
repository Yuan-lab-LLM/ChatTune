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
import aiohttp
import json
import pandas as pd
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Quality test with Customizable Parameters')
    parser.add_argument("--host", type=str, default="127.0.0.1", help="The inference server ip")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument('--medical-record', type=str, default="medical_record_data.json",
                        help='medical_record file name')
    parser.add_argument('--quality-file', type=str,
                        default="../data/raw/json/quality/quality.json", help='select quality file')
    parser.add_argument('--quality-name', type=str, default=None, help='select quality name')
    parser.add_argument('--output-file', type=str, default="output.xlsx",
                        help='output quality check file name')
    parser.add_argument("--batchsize", type=int, default=10)
    args = parser.parse_args()
    return args


async def process_single_data(session, data, args, url, headers):
    try:
        if args.quality_name is not None:
            data["input"]["control_quality_config_name"] = args.quality_name
        json_data = json.dumps(data, ensure_ascii=True)

        print(f"***lmx*** json_data {json_data} , data {data}")
        async with session.post(url, headers=headers, data=json_data, timeout=50) as response:
            response_text = await response.text()
            return response.status, response_text, data
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None, None


async def process_data_in_batches(data_list, args, batch_size, url, headers):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for data in data_list:
            task = process_single_data(session, data, args, url, headers)
            tasks.append(task)
            if len(tasks) == batch_size:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                tasks = []
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
    return results


def process_response_data(results):
    all_rows = []
    headers_excel = ['response_status_code', 'response_text', 'content', 'field', 'item', 'standard', 'check_quality',
                     'auto_modify_type', 'auto_modify_info', 'check_quality_detaile','amend_advice', "field_path", "error_segment_list"]
    input_base_medical_head = ['chief_complaint', 'history_of_present_illness', 'past_medical_history',
                               'personal_history', 'allergy_history', 'physical_examination', 'auxiliary_examination']
    input_physical_examination_head = ['temperature', 'pulse', 'blood_pressure', 'respiration']

    for status, text, data in results:
        if status is None or text is None:
            continue
        response_data = json.loads(text)
        print(f"***lmx*** response_data {json.dumps(response_data, ensure_ascii=False)}")

        # 写入响应状态和文本
        row = [status, text] + [None] * (len(headers_excel) - 2)
        all_rows.append(row)

        # 写入基础医疗记录信息
        response_base_medical = response_data['output']['basic_medical_record']
        for base_info in input_base_medical_head:
            row = [None] * 2 + [base_info, str(response_base_medical[base_info])] + [None] * (len(headers_excel) - 4)
            all_rows.append(row)

        # 写入体格检查信息
        response_physical_examination = response_data['output']['basic_medical_record']['physical_examination']
        for base_info in input_physical_examination_head:
            row = [None] * 2 + [base_info, response_physical_examination[base_info]] + [None] * (len(headers_excel) - 4)
            all_rows.append(row)

        # 写入质量检查信息
        response_quality_list = response_data['output']['control_quality']
        for response_info in response_quality_list:
            row = [None] * 2 + [
                response_info['content'],
                response_info['field'],
                response_info['item'],
                response_info['standard'],
                response_info['check_quality'],
                response_info['auto_modify_type'],
                response_info['auto_modify_info'],
                response_info['check_quality_detaile'],
                response_info['amend_advice'],
                response_info['field_path'],
                response_info['error_segment_list']
            ]
            all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=headers_excel)
    return df


if __name__ == "__main__":
    args = args_parser()

    # 读取 JSON 文件
    with open(args.medical_record, 'r', encoding='utf-8') as json_file:
        data_list = json.load(json_file)

    with open(args.quality_file, 'r', encoding='utf-8') as json_file:
        quality_info = json.load(json_file)
        quality_list = quality_info['check_quality']
        print(f"***lmx*** quality_list {quality_info} , quality_list {quality_list}")

    url = f"http://{args.host}:{args.port}/quality_inspect"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    # 并发大小
    batch = args.batchsize

    # 使用 asyncio.run 运行异步任务
    results = asyncio.run(process_data_in_batches(data_list, args, batch, url, headers))

    df = process_response_data(results)
    df.to_excel(args.output_file, index=False)