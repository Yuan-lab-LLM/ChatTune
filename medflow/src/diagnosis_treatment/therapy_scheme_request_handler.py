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

import copy
from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, String, insert, select, text)
from .base_diagnosis_request_handler import BaseDiagnosisRequestHandler
from .prompt_template import *
from .util_data_models import *
from .util_sqlite_function import *
from .util import *
import io
import time
import threading
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError
from fastapi import HTTPException

class TherapySchemeProcessChecker:
    def __init__(self) -> None:
        pass

    def check(self):
        return 6

class TherapySchemeRequestHandler(BaseDiagnosisRequestHandler):
    def __init__(self,
                 receive,
                 args,
                 scheme : None, 
                 sub_scheme : None,
                 request_type: None,
                 enable_think: False
                 ):
        super().__init__(receive, args, scheme, sub_scheme, request_type, enable_think)
        try:
            self.receive = RequestV6(**receive)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        self.db_engine = create_engine(f"sqlite:///{self.args.database}/medical_assistant.db")
        match scheme:
            case "generate_therapy":
                self.temprature = 1
                self.top_p = 1
                self.therapy_fields_reversed = {item.field_name: item.field_id for item in self.receive.input.therapy_fields}

    def checker_flag(self):
        self.checker = TherapySchemeProcessChecker()
        self.flag = self.checker.check()

    def generate_prompt(self, therapy_name, scheme):
        prompt = get_prompt('PromptScheme', self.receive, scheme, self.sub_scheme, therapy_name)
        prompt.set_prompt()
        return prompt

    def postprocess(self, answer):
        scheme_error = "Invalid Parameter: scheme must be within pick_therapy, generate_therapy."
        sema = threading.Semaphore(6)
        lock = threading.Lock()
        thread_list = []
        match self.scheme:
            case "pick_therapy":
                therapy_name = None
                prompt = self.generate_prompt(therapy_name, self.scheme)
                messages = self.preprocess(self.receive, prompt, self.flag)
                answer = self.predict(messages, self.temprature, self.top_p)
                self.results = self.postprocess_sm(self.receive, answer, self.scheme, self.sub_scheme, therapy_name)
            case "generate_therapy":
                scheme_num = len(self.receive.output.pick_therapy)
                sub_scheme_error = f"Invalid Parameter: sub_scheme must be within [1, {scheme_num}]."
                def infer_func(item):
                    sema.acquire()
                    try:
                        if item.therapy_name in self.therapy_fields_reversed:
                            therapy_name = self.therapy_fields_reversed[item.therapy_name]
                            if getattr(self.receive.output.generate_therapy[0].therapy_content, therapy_name) == list():
                                prompt = self.generate_prompt(therapy_name, self.scheme)
                                messages = self.preprocess(self.receive, prompt, self.flag)
                                answer = self.predict(messages, self.temprature, self.top_p, False)
                                with lock:
                                    self.results = self.postprocess_sm(self.receive, answer, self.scheme, self.sub_scheme, therapy_name)
                            match therapy_name:
                                case "prescription":
                                    if self.enable_think:
                                        prompt = self.generate_prompt(None, "rectify_therapy_prescription")
                                        messages = self.preprocess(self.receive, prompt, self.flag)
                                        answer = self.predict(messages, self.temprature, self.top_p)
                                        self.results = self.postprocess_sm(self.receive, answer, "rectify_therapy_prescription", self.sub_scheme, None)
                                case "transfusion":
                                    if self.enable_think:
                                        prompt = self.generate_prompt(None, "rectify_therapy_transfusion")
                                        messages = self.preprocess(self.receive, prompt, self.flag)
                                        answer = self.predict(messages, self.temprature, self.top_p)
                                        self.results = self.postprocess_sm(self.receive, answer, "rectify_therapy_transfusion", self.sub_scheme, None)
                        else:
                            self.results = self.receive
                            print(f"Error: There is no matcing therapy_name {item.therapy_name}.")
                    finally:
                        sema.release()
                if int(self.sub_scheme) in range(1, scheme_num+1, 1):
                    for item in self.receive.output.pick_therapy[int(self.sub_scheme) - 1].therapy_interpret:
                        t = threading.Thread(target=infer_func, args=(item,))
                        thread_list.append(t)
                        t.start()
                    for t in thread_list:
                        t.join()
                else:
                    raise HTTPException(status_code=400, detail=sub_scheme_error)
            case _:
                raise HTTPException(status_code=400, detail=scheme_error)
        results_dict = self.results.model_dump()
        results_json = json.dumps(results_dict, ensure_ascii=False)
        results = str(results_json)
        results = results.encode('utf-8')
        results = io.BytesIO(results)
        for re in results:
            yield re
        return

    def postprocess_sm(self, receive, answer, scheme, sub_scheme, therapy_name):
        params = receive
        hc = params.chat.historical_conversations
        hc_bak = params.chat.historical_conversations_bak
        if hc != [] and hc[-1].role == 'user':
            hc_bak.append(HistoricalConversations(role='user', content=hc[-1].content))
        hc.append(HistoricalConversations(role='assistant', content=answer))
        hc_bak.append(HistoricalConversations(role='assistant', content=answer))

        json_match, text_match = extract_json_and_text(answer)
        if not isinstance(json_match, re.Match):
            return params
        else:
            try:
                json_data = json_match.group(0)
                json_data = eval(json_data)
                # print(f"大模型匹配内容: \n{json_data=}\n")
            except:
                print("Error: There is no matching json data.")
                return params

        if scheme == "pick_therapy":
            pick_therapy_item = []
            if '治疗方案' in json_data and json_data['治疗方案']:
                for item in json_data['治疗方案']:
                    pick_therapy_sub_item = []
                    for sub_item in item['方案描述']:
                        pick_therapy_sub_item.append(TherapyInterpret(
                            therapy_name=sub_item['方案名称'],
                            therapy_content=sub_item['方案内容']
                        ))
                    pick_therapy_item.append(PickTherapy(
                        therapy_name=item['方案编号'],
                        therapy_summary=item['方案总述'],
                        therapy_interpret=pick_therapy_sub_item
                    ))
                params.output.pick_therapy = pick_therapy_item

        elif scheme == "generate_therapy":
            setattr(params.output.generate_therapy[0], "therapy_name", "方案"+sub_scheme)
            #if '处方' in json_data and json_data['处方']:
            #    prescription_item = []
            #    for item in json_data['处方']:
            #        query_item = item['药品名称']
            #        query_result = query_fastbm25(self.args.fastbm25_path, query_item, "prescription")
            #        if self.args.fastbm25 and query_result:
            #            query_item = query_result[0][0]
            #            sql_str = f"SELECT id, drug_code, drug_name, drug_specification, pharmacy_unit, \
            #                drug_formulation, COALESCE(alias, drug_name) AS name FROM prescription_info \
            #                WHERE name=\"{query_item}\""
            #            search_result = search_database(self.db_engine, sql_str)
            #            prescription_item.append(PrescriptionContent(
            #                drug_id=search_result[0][1],
            #                drug_name=item['药品名称'],
            #                drug_name_retrieve=search_result[0][2],
            #                drug_specification=search_result[0][3],
            #                manufacturer_name=item['厂家名称'],
            #                order_quantity=str(item['开单数量']),
            #                order_unit=item['开单单位'],
            #                route_of_administration=search_result[0][5],
            #                dosage=str(item['单次剂量']) if '单次剂量' in item.keys() else "",
            #                duration=item['持续天数'],
            #                frequency=item['用药频次'],
            #                corresponding_diseases=item['针对疾病'],
            #                drug_efficacy=item['药品作用']
            #            ))
            #                #order_unit=search_result[0][4],
            #        else:
            #            prescription_item.append(PrescriptionContent(
            #                drug_id=item['药品编号'],
            #                drug_name=item['药品名称'],
            #                drug_name_retrieve="",
            #                drug_specification=item['药品规格'],
            #                manufacturer_name=item['厂家名称'],
            #                order_quantity=str(item['开单数量']),
            #                order_unit=item['开单单位'],
            #                route_of_administration=item['用药途径'],
            #                dosage=str(item['单次剂量']) if '单次剂量' in item.keys() else "",
            #                duration=item['持续天数'],
            #                frequency=item['用药频次'],
            #                corresponding_diseases=item['针对疾病'],
            #                drug_efficacy=item['药品作用']
            #            ))
            #    setattr(params.output.generate_therapy[0].therapy_content, "prescription", prescription_item)

            if '处方' in json_data and json_data['处方']:
                prescription_item = []
                for item in json_data['处方']:
                    query_item = item['药品名称']
                    query_result = query_fastbm25(self.args.fastbm25_path, query_item, "prescription")
                    if self.args.fastbm25 and query_result:
                        query_item = query_result[0][0]
                        sql_str = f"SELECT id, drug_code, drug_name, drug_specification, pharmacy_unit, \
                            drug_formulation, COALESCE(alias, drug_name) AS name FROM prescription_info \
                            WHERE name=\"{query_item}\""
                        search_result = search_database(self.db_engine, sql_str)
                        prescription_item.append(PrescriptionContent(
                            drug_id=search_result[0][1],
                            drug_name=item['药品名称'],
                            drug_name_retrieve=search_result[0][2],
                            drug_specification=search_result[0][3],
                            manufacturer_name="",
                            order_quantity="",
                            order_unit="",
                            route_of_administration=search_result[0][5],
                            dosage="",
                            duration="",
                            frequency="",
                            corresponding_diseases=item['针对疾病'],
                            drug_efficacy=item['药品作用']
                        ))
                            #order_unit=search_result[0][4],
                    else:
                        prescription_item.append(PrescriptionContent(
                            drug_id="",
                            drug_name=item['药品名称'],
                            drug_name_retrieve="",
                            drug_specification=item['药品规格'],
                            manufacturer_name="",
                            order_quantity="",
                            order_unit="",
                            route_of_administration=item['用药途径'],
                            dosage="",
                            duration="",
                            frequency="",
                            corresponding_diseases=item['针对疾病'],
                            drug_efficacy=item['药品作用']
                        ))
                            #drug_id=item['药品编号'],
                setattr(params.output.generate_therapy[0].therapy_content, "prescription", prescription_item)

            #if '输液' in json_data and json_data['输液']:
            #    transfusion_item = []
            #    for item in json_data['输液']:
            #        query_item = item['药品名称']
            #        query_result = query_fastbm25(self.args.fastbm25_path, query_item, "prescription")
            #        if self.args.fastbm25 and query_result:
            #            query_item = query_result[0][0]
            #            sql_str = f"SELECT id, drug_code, drug_name, drug_specification, pharmacy_unit, \
            #                drug_formulation, COALESCE(alias, drug_name) AS name FROM prescription_info \
            #                WHERE name=\"{query_item}\""
            #            search_result = search_database(self.db_engine, sql_str)
            #            transfusion_item.append(TransfusionContent(
            #                drug_id=search_result[0][1],
            #                drug_name=item['药品名称'],
            #                drug_name_retrieve=search_result[0][2],
            #                drug_specification=search_result[0][3],
            #                manufacturer_name=item['厂家名称'],
            #                order_quantity=str(item['开单数量']),
            #                order_unit=item['开单单位'],
            #                route_of_administration=search_result[0][5],
            #                dosage=str(item['单次剂量']) if '单次剂量' in item.keys() else "",
            #                duration=item['持续天数'],
            #                frequency=item['用药频次'],
            #                corresponding_diseases=item['针对疾病'],
            #                drug_efficacy=item['药品作用'],
            #                infusion_group=item['输液分组'] if '输液分组' in item.keys() else "",
            #                infusion_rate=item['输液速度'] if '输液速度' in item.keys() else ""
            #            ))
            #                #order_unit=search_result[0][4],
            #        else:
            #            transfusion_item.append(TransfusionContent(
            #                drug_id=item['药品编号'],
            #                drug_name=item['药品名称'],
            #                drug_name_retrieve="",
            #                drug_specification=item['药品规格'],
            #                manufacturer_name=item['厂家名称'],
            #                order_quantity=str(item['开单数量']),
            #                order_unit=item['开单单位'],
            #                route_of_administration=item['用药途径'],
            #                dosage=str(item['单次剂量']) if '单次剂量' in item.keys() else "",
            #                duration=item['持续天数'],
            #                frequency=item['用药频次'],
            #                corresponding_diseases=item['针对疾病'],
            #                drug_efficacy=item['药品作用'],
            #                infusion_group=item['输液分组'] if '输液分组' in item.keys() else "",
            #                infusion_rate=item['输液速度'] if '输液速度' in item.keys() else ""
            #            ))
            #    setattr(params.output.generate_therapy[0].therapy_content, "transfusion", transfusion_item)
            if '输液' in json_data and json_data['输液']:
                transfusion_item = []
                for item in json_data['输液']:
                    query_item = item['药品名称']
                    query_result = query_fastbm25(self.args.fastbm25_path, query_item, "prescription")
                    if self.args.fastbm25 and query_result:
                        query_item = query_result[0][0]
                        sql_str = f"SELECT id, drug_code, drug_name, drug_specification, pharmacy_unit, \
                            drug_formulation, COALESCE(alias, drug_name) AS name FROM prescription_info \
                            WHERE name=\"{query_item}\""
                        search_result = search_database(self.db_engine, sql_str)
                        transfusion_item.append(TransfusionContent(
                            drug_id=search_result[0][1],
                            drug_name=item['药品名称'],
                            drug_name_retrieve=search_result[0][2],
                            drug_specification=search_result[0][3],
                            manufacturer_name="",
                            order_quantity="",
                            order_unit="",
                            route_of_administration=search_result[0][5],
                            dosage="",
                            duration="",
                            frequency="",
                            corresponding_diseases=item['针对疾病'],
                            drug_efficacy=item['药品作用'],
                            infusion_group="",
                            infusion_rate=""
                        ))
                            #order_unit=search_result[0][4],
                    else:
                        transfusion_item.append(TransfusionContent(
                            drug_id="",
                            drug_name=item['药品名称'],
                            drug_name_retrieve="",
                            drug_specification=item['药品规格'],
                            manufacturer_name="",
                            order_quantity="",
                            order_unit="",
                            route_of_administration=item['用药途径'],
                            dosage="",
                            duration="",
                            frequency="",
                            corresponding_diseases=item['针对疾病'],
                            drug_efficacy=item['药品作用'],
                            infusion_group="",
                            infusion_rate=""
                        ))
                setattr(params.output.generate_therapy[0].therapy_content, "transfusion", transfusion_item)

            if '处置' in json_data and json_data['处置']:
                disposition_item = []
                for item in json_data['处置']:
                    disposition_item.append(DispositionContent(
                        disposition_id=item['项目编号'],
                        disposition_name=item['项目名称'],
                        disposition_name_retrieve="",
                        frequency=item['频次'],
                        dosage=str(item['单次用量']) if '单次用量' in item.keys() else "",
                        duration=item['持续时间']
                    ))
                setattr(params.output.generate_therapy[0].therapy_content, "disposition", disposition_item)

            if '检查' in json_data and json_data['检查']:
                examine_item = []
                for item in json_data['检查']:
                    diagnose_item = []
                    for dia in item['针对疾病']:
                        query_item2 = dia['诊断名称']
                        query_result2 = query_fastbm25(self.args.fastbm25_path, query_item2, "diagnosis")
                        if self.args.fastbm25 and query_result2:
                            query_item2 = query_result2[0][0]
                            sql_str2 = f"SELECT id, diagnosis_code, diagnosis_name, COALESCE(alias, diagnosis_name) \
                                AS name FROM diagnosis_info WHERE name=\"{query_item2}\""
                            search_result2 = search_database(self.db_engine, sql_str2)
                            diagnose_item.append(Diagnosis(
                                diagnosis_name=dia['诊断名称'],
                                diagnosis_name_retrieve=search_result2[0][2],
                                diagnosis_code=search_result2[0][1],
                                diagnosis_identifier="疑似"
                            ))
                        else:
                            diagnose_item.append(Diagnosis(
                                diagnosis_name=dia['诊断名称'],
                                diagnosis_name_retrieve="",
                                diagnosis_code="",
                                diagnosis_identifier="疑似"
                            ))

                    query_item = item['检查名称']
                    query_result = query_fastbm25(self.args.fastbm25_path, query_item, "examination")
                    if self.args.fastbm25 and query_result:
                        query_item = query_result[0][0]
                        sql_str = f"SELECT id, examination_code, examination_name, examination_category, \
                            COALESCE(alias, examination_name) AS name FROM examination_info WHERE name=\"{query_item}\""
                        search_result = search_database(self.db_engine, sql_str)
                        examine_item.append(ExamineContent(
                            examine_code=search_result[0][1],
                            examine_category=search_result[0][3],
                            examine_name=item['检查名称'],
                            examine_name_retrieve=search_result[0][2],
                            order_quantity=str(item['开单数量']),
                            examine_result="",
                            corresponding_diagnosis=diagnose_item
                        ))
                    else:
                        examine_item.append(ExamineContent(
                            examine_code=item['检查编号'],
                            examine_category=item['检查类型'],
                            examine_name=item['检查名称'],
                            examine_name_retrieve="",
                            order_quantity=str(item['开单数量']),
                            examine_result="",
                            corresponding_diagnosis=diagnose_item
                        ))
                setattr(params.output.generate_therapy[0].therapy_content, "examine", examine_item)

            if '化验' in json_data and json_data['化验']:
                assay_item = []
                for item in json_data['化验']:
                    diagnose_item = []
                    for dia in item['针对疾病']:
                        query_item2 = dia['诊断名称']
                        query_result2 = query_fastbm25(self.args.fastbm25_path, query_item2, "diagnosis")
                        if self.args.fastbm25 and query_result2:
                            query_item2 = query_result2[0][0]
                            sql_str2 = f"SELECT id, diagnosis_code, diagnosis_name, COALESCE(alias, diagnosis_name) \
                                AS name FROM diagnosis_info WHERE name=\"{query_item2}\""
                            search_result2 = search_database(self.db_engine, sql_str2)
                            diagnose_item.append(Diagnosis(
                                diagnosis_name=dia['诊断名称'],
                                diagnosis_name_retrieve=search_result2[0][2],
                                diagnosis_code=search_result2[0][1],
                                diagnosis_identifier="疑似"
                            ))
                        else:
                            diagnose_item.append(Diagnosis(
                                diagnosis_name=dia['诊断名称'],
                                diagnosis_name_retrieve="",
                                diagnosis_code="",
                                diagnosis_identifier="疑似"
                            ))

                    query_item = item['项目名称']
                    query_result = query_fastbm25(self.args.fastbm25_path, query_item, "laboratorytest")
                    if self.args.fastbm25 and query_result:
                        query_item = query_result[0][0]
                        sql_str = f"SELECT id, laboratorytest_code, laboratorytest_name, laboratorytest_catagory, \
                            COALESCE(alias, laboratorytest_name) AS name FROM laboratorytest_info \
                            WHERE name=\"{query_item}\""
                        search_result = search_database(self.db_engine, sql_str)
                        assay_item.append(AssayContent(
                            assay_code=search_result[0][1],
                            assay_category=search_result[0][3],
                            assay_name=item['项目名称'],
                            assay_name_retrieve=search_result[0][2],
                            order_quantity=str(item['开单数量']),
                            assay_result=[],
                            corresponding_diagnosis=diagnose_item
                        ))
                    else:
                        assay_item.append(AssayContent(
                            assay_code=item['项目编号'],
                            assay_category=item['项目类型'],
                            assay_name=item['项目名称'],
                            assay_name_retrieve="",
                            order_quantity=str(item['开单数量']),
                            assay_result=[],
                            corresponding_diagnosis=diagnose_item
                        ))
                setattr(params.output.generate_therapy[0].therapy_content, "assay", assay_item)

            if '治疗方案' in json_data and json_data['治疗方案']:
                method_item = []
                for item in json_data['治疗方案']:
                    method_item.append(MethodTherapyContent(
                        method_code=item['治疗编号'],
                        method_type=item['治疗类型'],
                        method_name=item['治疗名称'],
                        method_name_retrieve="",
                        corresponding_diseases=item['针对疾病'],
                        method_plan=item['治疗计划'],
                        method_risk=item['潜在风险']
                    ))
                setattr(params.output.generate_therapy[0].therapy_content, therapy_name, method_item)

        elif scheme == "rectify_therapy_prescription":
            former_prescription = params.output.generate_therapy[0].therapy_content.model_dump()['prescription']
            new_prescription = json_data['处方']
            former_drug_name = {item['drug_name_retrieve'] if item['drug_name_retrieve'] else item['drug_name']: item 
                                for item in former_prescription}
            new_drug_name = {item['药品名称']: item for item in new_prescription}
            fields_to_replace = {
                "order_unit":"开单单位",
                "order_quantity":"开单数量",
                "dosage":"单次剂量",
                "duration":"持续天数",
                "frequency":"用药频次"
            }
            prescription_item = []
            for key, value in former_drug_name.items():
                if key in new_drug_name.keys():
                    for en, zh in fields_to_replace.items():
                        former_drug_name[key][en] = new_drug_name[key][zh]
                    prescription_item.append(PrescriptionContent.parse_obj(former_drug_name[key]))
            setattr(params.output.generate_therapy[0].therapy_content, "prescription", prescription_item)

        elif scheme == "rectify_therapy_transfusion":
            former_transfusion = params.output.generate_therapy[0].therapy_content.model_dump()['transfusion']
            new_transfusion = json_data['输液']
            former_drug_name = {item['drug_name_retrieve'] if item['drug_name_retrieve'] else item['drug_name']: item 
                                for item in former_transfusion}
            new_drug_name = {item['药品名称']: item for item in new_transfusion}
            fields_to_replace = {
                "order_unit":"开单单位",
                "order_quantity":"开单数量",
                "dosage":"单次剂量",
                "duration":"持续天数",
                "frequency":"用药频次",
                "infusion_group":"输液分组",
                "infusion_rate":"输液速度"
            }
            transfusion_item = []
            for key, value in former_drug_name.items():
                if key in new_drug_name.keys():
                    for en, zh in fields_to_replace.items():
                        former_drug_name[key][en] = new_drug_name[key][zh]
                    transfusion_item.append(TransfusionContent.parse_obj(former_drug_name[key]))
            setattr(params.output.generate_therapy[0].therapy_content, "transfusion", transfusion_item)

        return params