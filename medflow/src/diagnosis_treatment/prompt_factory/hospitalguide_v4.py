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

from ..prompt_template import *
from sqlalchemy import text
from ..util_sqlite_function import department_introduction
from fastapi import HTTPException

@register_prompt
class PromptHospitalGuide_v4(PromptTemplate):
    def __init__(self, receive, db_engine, scheme, department_path) -> None:
        super().__init__()
        self.yaml_name = "hospitalguide_v4.yaml"
        self.prompt_manager = PromptManager(self.yaml_name)
        self.ci_p = receive.input.client_info[0].patient
        if int(self.ci_p.patient_age) < 6:
            self.all_department = list(department.department_name for department in receive.input.all_department if department.if_child == 1) \
                if receive.input.all_department != None else ""
            self.department_intro = department_introduction(department_path, if_child=1)
        else:
            self.all_department = list(department.department_name for department in receive.input.all_department if department.if_child == 0) \
                if receive.input.all_department != None else ""
            self.department_intro = department_introduction(department_path)
        self.bmr = receive.output.basic_medical_record
        self.db_engine = db_engine
        self.scheme = scheme

    def set_prompt(self):
        self.variables = {
            "patient_name": self.ci_p.patient_name,
            "patient_gender": self.ci_p.patient_gender,
            "patient_age": self.ci_p.patient_age,
            "reversed_patient_gender": gender_map.get(self.ci_p.patient_gender),
            "format_hospital_guide1": format_hospital_guide1,
            "format_hospital_guide2": format_hospital_guide2,
            "all_department": self.all_department,
            "department_intro": self.department_intro,
            "chief_complaint": self.bmr.chief_complaint,
            "history_of_present_illness": self.bmr.history_of_present_illness,
            "past_medical_history": self.bmr.past_medical_history,
            "personal_history": self.bmr.personal_history,
            "allergy_history": self.bmr.allergy_history,
            "format_basic_medical_record": format_basic_medical_record
        }
        self.prompt = {
            "81": self.__set_hospital_guide(),
            "8": self.__set_modify_record()
        }
        return self.prompt

    def __set_hospital_guide(self):
        match self.scheme:
            case "simple":
                system_str = self.prompt_manager.get_prompt("simple", 0, self.variables)
            case "detailed":
                system_str = self.prompt_manager.get_prompt("detailed", 0, self.variables)
                if int(self.ci_p.patient_age) >= 18:
                    system_str += self.prompt_manager.get_prompt("detailed", 1, self.variables)
                else:
                    system_str += self.prompt_manager.get_prompt("detailed", 2, self.variables)
                system_str += self.prompt_manager.get_prompt("detailed", 3, self.variables)
            case _:
                raise HTTPException(status_code=400, detail=f"Invalid Parameter: scheme must be equal to simple or detailed.")
        return system_str, None

    def __set_modify_record(self):
        system_str = self.prompt_manager.get_prompt("modify_reocrd", 0, self.variables)
        return system_str, None

#    def search_department_all(self):
#        sql_str = f"SELECT department_name FROM department_info WHERE department_hierarchy_code LIKE '%.%.' \
#OR department_hierarchy_code LIKE '%.%.%'"
#        with self.db_engine.connect() as con:
#            rows = con.execute(text(sql_str))
#            sub_dept = [row[0] for row in rows]
#        return sub_dept
