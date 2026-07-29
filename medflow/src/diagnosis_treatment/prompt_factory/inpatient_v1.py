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
import json

from ..prompt_template import (
    PromptManager,
    PromptTemplate,
    format_inpatient_map,
    inpatient_fields,
    register_prompt,
    reversed_inpatient_fields,
)


@register_prompt
class PromptInPatient_v1(PromptTemplate):
    def __init__(self, receive, scheme) -> None:
        super().__init__()
        self.input = receive.input
        self.yaml_name = "inpatient_v1.yaml"
        self.prompt_manager = PromptManager(self.yaml_name)
        self.scheme = scheme

    def __dynamic_inpatient(self):
        dyn_inpatient = json.loads(self.input.inpatient.json())
        if "physical_examination" in dyn_inpatient:
            dyn_inpatient["physical_examination"] = {reversed_inpatient_fields.get(key): value 
                for key, value in dyn_inpatient["physical_examination"].items()}
        diagnosis = list(filter(lambda k: "diagnosis" in k, dyn_inpatient.keys()))
        for item in diagnosis:
            dyn_inpatient[item] = [{reversed_inpatient_fields.get(key): value for key, value in item.items()
                if key not in ["diagnosis_name_retrieve", "diagnosis_code"]} for item in dyn_inpatient[item]]
        dyn_inpatient = {reversed_inpatient_fields.get(key): value for key, value in dyn_inpatient.items()}
        for key in list(dyn_inpatient):
            if inpatient_fields.get(key) not in self.input.included_fields:
                del dyn_inpatient[key]
        return dyn_inpatient

    def __dynamic_format_inpatient(self):
        dyn_format_inpatient = copy.copy(format_inpatient_map.get(f"format_{self.scheme}"))
        for key in list(dyn_format_inpatient):
            if inpatient_fields.get(key) not in self.input.included_fields:
                del dyn_format_inpatient[key]
        return dyn_format_inpatient

    def set_prompt(self):
        dyn_inpatient = self.__dynamic_inpatient()
        dyn_format_inpatient = self.__dynamic_format_inpatient()
        self.variables = {
            "inpatient": dyn_inpatient,
            "doctor_supplement": self.input.doctor_supplement,
            "format_inpatient": dyn_format_inpatient,
        }
        self.prompt = {self.scheme: self.__set_inpatient()}
        return self.prompt

    def __set_inpatient(self):
        system_str = self.prompt_manager.get_prompt("inpatient", 0, self.variables)
        user_str = None
        return system_str, user_str
