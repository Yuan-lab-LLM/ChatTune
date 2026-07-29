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
import io
import json
import re

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine

from .base_diagnosis_request_handler import BaseDiagnosisRequestHandler
from .prompt_template import get_prompt, inpatient_fields
from .util import extract_json_and_text
from .util_data_models import (
    Diagnosis,
    HistoricalConversations,
    PhyscialExamination,
    RequestAdmissionRecord,
    RequestDischarge,
    RequestFirstPage,
    RequestInformedConsentForm,
    RequestNotification,
    RequestProgressNote,
    RequestSurgicalRecord,
)
from .util_sqlite_function import query_fastbm25, search_database


class InPatientRequestHandler(BaseDiagnosisRequestHandler):
    def __init__(
        self,
        receive,
        args,
        scheme: None,
        sub_scheme: None,
        request_type: None,
        enable_think: False,
    ):
        super().__init__(receive, args, scheme, sub_scheme, request_type, enable_think)
        self.request_map = {
            "admission_record": RequestAdmissionRecord,
            "first_page": RequestFirstPage,
            "progress_note": RequestProgressNote,
            "surgical_record": RequestSurgicalRecord,
            "informed_consent": RequestInformedConsentForm,
            "notification": RequestNotification,
            "discharge_summary": RequestDischarge,
            "discharge_record": RequestDischarge,
        }
        try:
            if self.scheme in self.request_map:
                self.receive = self.request_map[self.scheme](**receive)
            else:
                inpatient_error = f"Invalid Parameter: scheme must be within {list(self.request_map)}"
                raise HTTPException(status_code=422, detail=inpatient_error)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        self.db_engine = create_engine(f"sqlite:///{self.args.database}/medical_assistant.db")

    def checker_flag(self):
        self.flag = self.scheme

    def generate_prompt(self):
        self.prompt = get_prompt("PromptInPatient", self.receive, self.scheme)
        self.prompt.set_prompt()

    def postprocess(self, answer):
        results = self.postprocess_dn(self.receive, answer)
        results_dict = results.model_dump()
        results_json = json.dumps(results_dict, ensure_ascii=False)
        results = str(results_json)
        results = results.encode("utf-8")
        results = io.BytesIO(results)
        for result in results:
            yield result
        return

    def postprocess_dn(self, receive, answer):
        params = copy.deepcopy(receive)
        hc = params.chat.historical_conversations
        hc_bak = params.chat.historical_conversations_bak
        if hc != [] and hc[-1].role == "user":
            hc_bak.append(HistoricalConversations(role="user", content=hc[-1].content))
        hc.append(HistoricalConversations(role="assistant", content=answer))
        hc_bak.append(HistoricalConversations(role="assistant", content=answer))

        json_match, text_match = extract_json_and_text(answer)
        if not isinstance(json_match, re.Match):
            return params
        try:
            json_data = json_match.group(0)
            json_data = eval(json_data)
            # print(f"匹配内容: \n{json_data=}\n")
        except Exception:
            print("Error: There is no matching json data.")
            return params

        def diagnosis(key):
            diagnosis = []
            for item in json_data[key]:
                bm25_results = query_fastbm25(self.args.fastbm25_path, item["诊断名称"], "diagnosis")
                if self.args.fastbm25 and bm25_results:
                    sql_sen = f'SELECT id, diagnosis_code, diagnosis_name, COALESCE(alias, diagnosis_name) \
                        AS name FROM diagnosis_info WHERE name="{bm25_results[0][0]}"'
                    sql_results = search_database(self.db_engine, sql_sen)
                    diagnosis.append(Diagnosis.parse_obj({
                        "diagnosis_name": item["诊断名称"],
                        "diagnosis_name_retrieve": sql_results[0][2],
                        "diagnosis_code": sql_results[0][1],
                        "diagnosis_identifier": item["诊断标识"],
                    }))
                else:
                    diagnosis.append(Diagnosis.parse_obj({
                        "diagnosis_name": item["诊断名称"],
                        "diagnosis_name_retrieve": "",
                        "diagnosis_code": "",
                        "diagnosis_identifier": item["诊断标识"],
                    }))
            return diagnosis

        if self.scheme in list(self.request_map):
            inpatient = params.output.inpatient
            included_fields = params.input.included_fields
            if "体格检查" in json_data:
                inpatient.physical_examination = PhyscialExamination.parse_obj({
                    inpatient_fields.get(key): value for key, value in json_data["体格检查"].items()
                })
            if "初步诊断" in json_data:
                inpatient.initial_diagnosis = diagnosis("初步诊断")
            if "主要诊断" in json_data:
                inpatient.principal_diagnosis = diagnosis("主要诊断")
            if "其他诊断" in json_data:
                inpatient.other_diagnosis = diagnosis("其他诊断")
            if "术中诊断" in json_data:
                inpatient.intraoperative_diagnosis = diagnosis("术中诊断")
            if "诊断" in json_data:
                inpatient.diagnosis = diagnosis("诊断")
            if "出院诊断" in json_data:
                inpatient.discharge_diagnosis = diagnosis("出院诊断")

            for key in list(json_data):
                r_key = inpatient_fields.get(key)
                if r_key in included_fields and r_key not in [
                    "physical_examination",
                    "initial_diagnosis",
                    "principal_diagnosis",
                    "other_diagnosis",
                    "intraoperative_diagnosis",
                    "diagnosis",
                    "discharge_diagnosis",
                ]:
                    setattr(inpatient, r_key, json_data[key])

        return params
