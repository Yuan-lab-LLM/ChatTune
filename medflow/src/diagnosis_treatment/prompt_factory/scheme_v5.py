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

import json
from ..prompt_template import *
from fastapi import HTTPException

@register_prompt
class PromptScheme_v5(PromptTemplate):
    def __init__(self, receive, scheme, sub_scheme, therapy_name) -> None:
        super().__init__()
        self.yaml_name = "scheme_v5.yaml"
        self.prompt_manager = PromptManager(self.yaml_name)
        self.scheme = scheme
        self.sub_scheme = sub_scheme
        self.ci_p = receive.input.client_info[0].patient
        self.bmr = receive.input.basic_medical_record
        self.diag = receive.input.diagnosis
        self.therapy_fields = {item.field_id: item.field_name for item in receive.input.therapy_fields}
        #self.diagnose_definite = [(item.diagnosis_name_retrieve or item.diagnosis_name)
        #    for item in self.diag if item.diagnosis_identifier == "确诊"]
        self.diagnose_definite = [(item.diagnosis_name_retrieve or item.diagnosis_name) for item in self.diag]
        self.pick_therapy = receive.output.pick_therapy
        physical_examination = json.loads(self.bmr.physical_examination.json())
        self.physical_examination = {reversed_sub_medical_fields.get(k): v for k, v in physical_examination.items()}
        self.output = receive.output
        self.therapy_name = therapy_name

    def set_prompt(self):
        self.variables = {
            "therapy_fields": "、".join(self.therapy_fields.values()),
            "format_pick_therapy": format_pick_therapy,
            "patient_name": self.ci_p.patient_name,
            "patient_gender": self.ci_p.patient_gender,
            "patient_age": self.ci_p.patient_age,
            "chief_complaint": self.bmr.chief_complaint,
            "history_of_present_illness": self.bmr.history_of_present_illness,
            "personal_history": self.bmr.personal_history,
            "allergy_history": self.bmr.allergy_history,
            "physical_examination": self.physical_examination,
            "auxiliary_examination": self.bmr.auxiliary_examination,
            "diagnose_definite": self.diagnose_definite,
            "format_prescription": format_prescription,
            "format_transfusion": format_transfusion,
            "format_disposition": format_disposition,
            "format_examine": format_examine,
            "format_assay": format_assay,
            "format_generate_therapy": format_generate_therapy
        }
        self.prompt = {
            "6": self.__set_default_therapy(),
        }
        return self.prompt

    def __therapy_interpret(self):
        if self.pick_therapy != []:
            therapy_interpret = [i.therapy_content for i in self.output.pick_therapy[int(self.sub_scheme)-1].therapy_interpret
               if i.therapy_name == self.therapy_fields[self.therapy_name]]
        else:
           therapy_interpret = ['无']
        return therapy_interpret 

    def __get_input_prescription(self):
        input_format = ""
        if self.scheme == "rectify_therapy_prescription":
            prescription = self.output.generate_therapy[0].therapy_content.prescription
            input_format = {"处方": []}
            for p in prescription:
                input_format["处方"].append({
                    "药品编号": p.drug_id,
                    "药品名称": p.drug_name_retrieve if p.drug_name_retrieve else p.drug_name,
                    "药品规格": p.drug_specification,
                    "用药途径": p.route_of_administration,
                    "开单单位": "",
                    "开单数量": "",
                    "单次剂量": "",
                    "持续天数": "",
                    "用药频次": "",
                    "针对疾病": p.corresponding_diseases
                })
                    #"开单单位": p.order_unit,
        if self.scheme == "rectify_therapy_transfusion":
            transfusion = self.output.generate_therapy[0].therapy_content.transfusion
            input_format = {"输液": []}
            for p in transfusion:
                input_format["输液"].append({
                    "药品编号": p.drug_id,
                    "药品名称": p.drug_name_retrieve if p.drug_name_retrieve else p.drug_name,
                    "药品规格": p.drug_specification,
                    "用药途径": p.route_of_administration,
                    "开单单位": "",
                    "开单数量": "",
                    "单次剂量": "",
                    "持续天数": "",
                    "用药频次": "",
                    "输液分组": "",
                    "输液速度": "",
                    "针对疾病": p.corresponding_diseases
                })
                    #"开单单位": p.order_unit,
        return str(input_format)

    def __get_output_prescription(self):
        output_format = ""
        if self.scheme == "rectify_therapy_prescription":
            output_format = {"处方": []}
            for i in range(3):
                output_format["处方"].append({
                    "药品编号": "",
                    "药品名称": "",
                    "药品规格": "",
                    "用药途径": "",
                    "开单单位": "",
                    "开单数量": "",
                    "单次剂量": "",
                    "持续天数": "",
                    "用药频次": "",
                    "针对疾病": ""
                })
        if self.scheme == "rectify_therapy_transfusion":
            output_format = {"输液": []}
            for i in range(3):
                output_format["输液"].append({
                    "药品编号": "",
                    "药品名称": "",
                    "药品规格": "",
                    "用药途径": "",
                    "开单单位": "",
                    "开单数量": "",
                    "单次剂量": "",
                    "持续天数": "",
                    "用药频次": "",
                    "输液分组": "",
                    "输液速度": "",
                    "针对疾病": ""
                })
        return str(output_format)
        
    def __set_default_therapy(self):
        scheme_error = "Invalid Parameter: scheme must be within pick_therapy, generate_therapy."
        therapy_fields_keys = ", ".join(self.therapy_fields.keys())
        therapy_name_error = f"Invalid Parameter: therapy_name must be within {therapy_fields_keys}."
        if self.scheme == "pick_therapy":
            system_str = self.prompt_manager.get_prompt(self.scheme, 0, self.variables)
            user_str = self.prompt_manager.get_prompt(self.scheme, 1, self.variables)
        elif self.scheme == "generate_therapy":
            if self.therapy_name in self.therapy_fields.keys():
                self.variables["picked_therapy"] = self.therapy_fields[self.therapy_name]
                self.variables["therapy_interpret"] = self.__therapy_interpret()
            else:
                raise HTTPException(status_code=400, detail=therapy_name_error)

            if self.therapy_name in ["prescription","transfusion","disposition","examine","assay"]:
                system_str = self.prompt_manager.get_prompt(self.therapy_name, 0, self.variables)
                user_str = self.prompt_manager.get_prompt(self.therapy_name, 1, self.variables)
            else:
                system_str = self.prompt_manager.get_prompt("therapy", 0, self.variables)
                user_str = self.prompt_manager.get_prompt("therapy", 1, self.variables)
        elif self.scheme in ["rectify_therapy_prescription", "rectify_therapy_transfusion"]:
            self.variables["input_format"] = self.__get_input_prescription()
            self.variables["output_format"] = self.__get_output_prescription()
            system_str = self.prompt_manager.get_prompt(self.scheme, 0, self.variables)
            user_str = self.prompt_manager.get_prompt(self.scheme, 1, self.variables)
        else:
            raise HTTPException(status_code=400, detail=scheme_error)
        return system_str, user_str