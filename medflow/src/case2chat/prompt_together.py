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
# chat format

chat_format=[
    {
        "role": "user",
        "content": ""
    },
    {
        "role": "assistant",
        "content": ""
    },
    {
        "role": "user",
        "content": ""
    },
    {
        "role": "assistant",
        "content": ""
    }
]

class PromptTemplate():
    """
    Prompt Templates.
    Convert Medical Records To Conversations.
    """
    def __init__(self, params):
        self.chief_complaint=params.chief_complaint
        self.history_of_present_illness=params.history_of_present_illness
        self.past_medical_history=params.past_medical_history
        self.personal_history=params.personal_history
        self.allergy_history=params.allergy_history
        self.physical_examination=params.physical_examination
        self.auxiliary_examination=params.auxiliary_examination
        
    def generate_prompt(self):
        # Prompt for converting medical records to conversations
        self.common=f"【主诉】为{self.chief_complaint}，【现病史】为{self.history_of_present_illness}，【既往史】为{self.past_medical_history}，【个人史】为{self.personal_history}，【过敏史】为{self.allergy_history}，【辅助检查】为{self.auxiliary_examination}"

        system_str=f"""##Role
医生
##obtrain
1.你拥有专业的医学知识，能够用简洁、专业的对话了解患者病情；
2.你每次与患者对话只能针对当前已知的病情提问一个最关键的问题；
3.你预先不知道患者的症状、既往史、过敏史及个人史；
4.患者不懂医学知识，表达口语化；
5.对话应从患者开始，到医生了解完所有症状、既往史、过敏史及个人史结束；
6.你禁止重复患者说过的话，同时也禁止反问患者已经提到的症状。
## workflow:
1.询问患者当前的症状，与患者进行多轮沟通，以尽快确诊。
2.向患者了解跟病情相关的既往史与慢性病史。
3.向患者了解是否有烟酒史。
4.向患者了解是否有过敏史。
## Initialization :
作为 <Role>，严格遵守<obtrain>中的规则，按照<workflow>的工作流程，对输入的病历模拟生成中文医患多轮对话数据，医生为assistant，患者为user，格式仿照\n{chat_format}。"""

        user_str=f"""病历中的{self.common}，请生成中文医患多轮对话数据。"""

        return system_str, user_str

