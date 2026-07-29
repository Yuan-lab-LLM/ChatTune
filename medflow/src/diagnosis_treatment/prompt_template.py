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
import json
import yaml
from jinja2 import Template
from fastapi import HTTPException

import collections
VersionedPromptFactory = collections.defaultdict(dict)

format_distribute="""[
    {
        "患者意图": ""
    }
]
"""

format_client_select="""{
    "就诊人姓名": ""
}"""

format_client_info="""{
    "档案": {
        "患者信息": {
            "姓名": "",
            "是否儿童": "",
            "性别": "",
            "年龄": "",
            "出生日期": {
                "年": "",
                "月": "",
                "日":""
            },
            "证件类型": "大陆居民身份证",
            "证件号码": "",
            "手机号码": "",
            "所居区域": {
                "省": "",
                "市": "",
                "区": "",
                "街道": ""
            },
            "详细地址": ""
        },
        "监护人信息": {
            "姓名": "",
            "证件类型": "",
            "证件号码": ""
        }
    }
}
"""

#病历
format_basic_medical_record="""{
    "主诉": "",
    "现病史": "",
    "既往史": "xxx,xxx,xxx",
    "个人史": "xxx;xxx",
    "过敏史": "xxx;xxx;xxx"
}
"""

#科室
format_department_single="""[
    {
        "科室编号": "001",
        "科室名称": ""
    }
]
"""
format_department_multi="""[
    {
        "科室编号": "001",
        "科室名称": ""
    },
    {
        "科室编号": "002",
        "科室名称": ""
    }
]
"""

#挂号
format_hospital_register="""[{
    "科室编号": "",
    "科室名称": "",
    "医生编号": "",
    "医生姓名": "",
    "医生职称": "",
    "挂号日期": "",
    "起始时间": "",
    "终止时间": "",
    "号源数量": "",
    }
]
"""

format_hospital_register_modify="""[{"科室名称": "","医生姓名": "","挂号日期": "","起始时间": "","终止时间": "","号源数量": ""}]"""

REGISTER_MODEL_TYPE_BASE = "register_type_base"
REGISTER_MODEL_TYPE_INTENTION = "register_type_intention"

register_intention_json_str = '''
{
    "意图类型": "",
    "科室名称": "",
    "医生姓名": "",
    "医生职称": "",
    "日期": "",
    "时间": "",
    "号源": "",
    "查询主体": "",
    "优先级": ""
}
'''

register_intention_field_mapping = {
    "意图类型": "intention_tpye",
    "科室名称": "department_name",
    "医生姓名": "doctor_name",
    "医生职称": "doctor_title",
    "日期": "register_date",
    "时间": "register_time",
    "号源": "register_source",
    "查询主体": "query_subject",
    "优先级": "priority"
}


#诊断
format_diagnose="""{
"诊断": [{
       "诊断名称": "痛风",
       "诊断编码": "M10.900",
       "诊断标识": "疑似"
    },
    {
       "诊断名称": "关节炎",
       "诊断编码": "M13.900",
       "诊断标识": "疑似"
    }]
}
"""

#检查和化验
format_examine_assay="""{
"检查": [{
       "检查编号": "",
       "检查类型": "",
       "检查名称": "",
       "针对疾病": [
           {
               "诊断名称": ""
           }
       ],
       "开单数量": "1"
    }],
"化验": [{
        "项目编号": "JY8110",
        "项目类型": "临检类",
        "项目名称": "血常规（五分类）",
        "针对疾病": [
            {
                "诊断名称": ""
            }
        ],
        "开单数量": "1"
    }]
}
"""
#检查
format_examine="""{
"检查": [{
       "检查编号": "",
       "检查类型": "",
       "检查名称": "",
       "针对疾病": [
           {
               "诊断名称": ""
           }
       ],
       "开单数量": "1"
    }]
}
"""
#化验
format_assay="""{
"化验": [{
        "项目编号": "JY8110",
        "项目类型": "临检类",
        "项目名称": "血常规（五分类）",
        "针对疾病": [
            {
                "诊断名称": ""
            }
        ],
        "开单数量": "1"
    }]
}
"""

#处方
format_prescription="""{
"处方": [{
    "药品名称": "维生素C片",
    "药品规格": "0.1g*100",
    "用药途径": "口服",
    "针对疾病": "",
    "药品作用": ""
},
{
    "药品名称": "",
    "药品规格": "0.1g*100",
    "用药途径": "口服",
    "针对疾病": "",
    "药品作用": ""
},
{
    "药品名称": "",
    "药品规格": "0.1g*100",
    "用药途径": "口服",
    "针对疾病": "",
    "药品作用": ""
}]}
"""

#输液
format_transfusion="""{
"输液": [{
    "药品名称": "维生素C注射液",
    "药品规格": "0.5g*10*2ml",
    "用药途径": "静脉滴注",
    "针对疾病": "",
    "药品作用": "",
    "输液分组": "第一组",
    "输液速度": "30gtt/min"
},
{
    "药品名称": "",
    "药品规格": "0.5g*10*2ml",
    "用药频次": "每日一次",
    "针对疾病": "",
    "药品作用": "",
    "输液分组": "第一组",
    "输液速度": "30gtt/min"
}]}
"""

#处置
format_disposition="""{
    "处置": [{
        "项目编号": "CZ123457",
        "项目名称": "换药",
        "频次": "Qd",
        "单次用量": "1",
        "持续时间": "1天"
    },
    {
        "项目编号": "CZ123457",
        "项目名称": "",
        "频次": "Qd",
        "单次用量": "1",
        "持续时间": "1天"
    }]
}
"""

#多方案-挑选
format_pick_therapy="""{
"治疗方案": [
    {
        "方案编号":"方案一",
        "方案描述": [
            {
                "方案名称":"",
                "方案内容":""
            },
            {
                "方案名称":"",
                "方案内容":""
            }
        ],
        "方案总述":""
    },
    {
        "方案编号":"方案二",
        "方案描述": [
            {
                "方案名称":"",
                "方案内容":""
            },
            {
                "方案名称":"",
                "方案内容":""
            }
        ],
        "方案总述":""
    }
]}
"""

#多方案-生成
format_generate_therapy="""{
    "治疗方案": [{
       "治疗编号": "001",
       "治疗类型": "",
       "治疗名称": "",
       "针对疾病": "",
       "治疗计划": "",
       "潜在风险": ""
    },
    {
       "治疗编号": "002",
       "治疗类型": "",
       "治疗名称": "",
       "针对疾病": "",
       "治疗计划": "",
       "潜在风险": ""
    }]
}
"""

#多方案-治疗
format_generate_medicine="""
{
"处方":[
{
"药品编号":"YP123456",
"药品名称":"维生素C片",
"药品规格":"0.1g*100",
"厂家名称":"XXXXXXXX制药厂",
"开单数量":"10",
"开单单位":"粒",
"用药途径":"口服",
"单次剂量":"1粒",
"持续天数":"7天",
"用药频次":"每日三次",
"针对疾病":"",
"药品作用":""
},
{
"药品编号":"YP123457",
"药品名称":"",
"药品规格":"0.1g*100",
"厂家名称":"XXXXXXXX制药厂",
"开单数量":"10",
"开单单位":"粒",
"用药途径":"口服",
"单次剂量":"1粒",
"持续天数":"7天",
"用药频次":"每日三次",
"针对疾病":"",
"药品作用":""
},
{
"药品编号":"YP123458",
"药品名称":"",
"药品规格":"0.1g*100",
"厂家名称":"XXXXXXXX制药厂",
"开单数量":"10",
"开单单位":"粒",
"用药途径":"口服",
"单次剂量":"1粒",
"持续天数":"7天",
"用药频次":"每日三次",
"针对疾病":"",
"药品作用":""
}
],
"输液":[
{
"药品编号":"ZS123456",
"药品名称":"维生素C注射液",
"药品规格":"0.5g*10*2ml",
"厂家名称":"XXXXXXXX制药股份有限公司",
"开单数量":"1",
"开单单位":"支",
"用药途径":"静脉滴注",
"单次剂量":"1支",
"持续天数":"1天",
"用药频次":"每日一次",
"针对疾病":"",
"药品作用":"",
"输液分组":"第一组",
"输液速度":"30gtt/min"
},
{
"药品编号":"ZS123457",
"药品名称":"",
"药品规格":"0.5g*10*2ml",
"厂家名称":"XXXXXXXX制药股份有限公司",
"开单数量":"1",
"开单单位":"支",
"用药途径":"静脉滴注",
"单次剂量":"1支",
"持续天数":"1天",
"用药频次":"每日一次",
"针对疾病":"",
"药品作用":"",
"输液分组":"第一组",
"输液速度":"30gtt/min"
}
],
"处置":[
{
"项目编号":"CZ123457",
"项目名称":"换药",
"频次":"Qd",
"单次用量":"1",
"持续时间":"1天"
},
{
"项目编号":"CZ123457",
"项目名称":"",
"频次":"Qd",
"单次用量":"1",
"持续时间":"1天"
}
]
}
"""


#复诊
format_return_visit="""{
    "病情总结": "",
    "是否复诊": ""
}
"""

#导诊
format_hospital_guide1="""{"病历": {"主诉": ""},"推荐科室": [{"科室编号": "001","科室名称": "","是否儿科": ""}]}
"""
format_hospital_guide2="""{"病历": {"主诉": ""},"推荐科室": [{"科室编号": "001","科室名称": "","是否儿科": ""},{"科室编号": "002","科室名称": "","是否儿科": ""}]}
"""

gender_map={"男": "女", "女": "男"}

format_translate = {"原句":"", "翻译结果":""}

medical_fields={
    "主诉":"chief_complaint",
    "现病史":"history_of_present_illness",
    "既往史":"past_medical_history",
    "个人史":"personal_history",
    "过敏史":"allergy_history",
    "体格检查":"physical_examination",
    "辅助检查":"auxiliary_examination",
    "专科检查":"specialty_examination",
    "治疗":"cure",
    "医嘱":"doctor_advice"
}

reversed_medical_fields={v: k for k, v in medical_fields.items()}

sub_medical_fields={
    "体温":"temperature",
    "脉搏":"pulse",
    "血压":"blood_pressure",
    "呼吸":"respiration"
}

reversed_sub_medical_fields={v: k for k, v in sub_medical_fields.items()}

request_type_map={
   "distribute": "v0",
   "clientinfo": "v1",
   "basicmedicalrecord": "v2",
   "hospitalregister": "v3",
   "diagnosis": "v4",
   "examass": "v5",
   "scheme": "v6",
   "returnvisit": "v7",
   "hospitalguide": "v8",
   "doctormedicalrecord": "v9",
   "inpatient": "inpatient",
}

format_admission_record = {
   "姓名": "",
   "性别": "",
   "年龄": "",
   "民族": "",
   "职业": "",
   "婚姻状况": "",
   "出生地": "",
   "入院时间": "",
   "记录时间": "",
   "病史陈述者": "",
   "主诉": "",
   "现病史": "",
   "既往史": "",
   "体格检查": {
      "体温": "",
      "脉搏": "",
      "血压": "",
      "呼吸": "",
   },
   "辅助检查": "",
   "医师签名": "",
   "初步诊断": [{"诊断名称": "", "诊断标识": ""}, {"诊断名称": "", "诊断标识": ""}]
}

format_first_page = {
   "医疗机构名称": "",
   "医疗机构代码": "",
   "姓名": "",
   "性别": "",
   "年龄": "",
   "病案号": "",
   "身份证号": "",
   "入院日期": "",
   "出院日期": "",
   "入院途径": "",
   "出院方式": "",
   "手术名称": "",
   "手术日期": "",
   "手术级别": "",
   "切口愈合等级": "",
   "医疗付费方式": "",
   "总费用": "",
   "主治医师签名": "",
   "责任护士签名": "",
   "编码员签名": "",
   "科主任签名": "",
   "主要诊断": [{"诊断名称": "", "诊断标识": ""}, {"诊断名称": "", "诊断标识": ""}],
   "其他诊断": [{"诊断名称": "", "诊断标识": ""}, {"诊断名称": "", "诊断标识": ""}]
}

format_progress_note = {
   "记录日期": "",
   "病情变化": "",
   "检查结果": "",
   "诊疗措施": "",
   "医嘱调整理由": "",
   "上级医师查房意见": "",
   "抢救记录": "",
   "医师签名": ""
}

format_surgical_record = {
   "手术名称": "",
   "手术日期": "",
   "手术开始时间": "",
   "手术结束时间": "",
   "术者姓名": "",
   "助手姓名": "",
   "麻醉方式": "",
   "麻醉医师姓名": "",
   "手术步骤": "",
   "出血量": "",
   "输血量": "",
   "器械敷料清点": "",
   "术后去向": "",
   "医师签名": "",
   "术中诊断": [{"诊断名称": "", "诊断标识": ""}, {"诊断名称": "", "诊断标识": ""}]
}

format_informed_consent = {
   "操作/手术名称": "",
   "预期获益": "",
   "可预见风险": "",
   "替代治疗方案": "",
   "风险应对措施": "",
   "患者声明": "",
   "患者签名": "",
   "医师签名": "",
   "签署日期": "",
}

format_notification = {
   "患者姓名": "",
   "病危/重依据": "",
   "预后风险": "",
   "告知医师签名": "",
   "接收方签名": "",
   "关系证明": "",
   "签署时间": "",
   "诊断": [{"诊断名称": "", "诊断标识": ""}, {"诊断名称": "", "诊断标识": ""}]
}

format_discharge_summary = {
   "姓名": "",
   "性别": "",
   "年龄": "",
   "住院号": "",
   "入院日期": "",
   "出院日期": "",
   "住院天数": "",
   "入院主诉": "",
   "重要检查结果": "",
   "手术/操作名称": "",
   "出院情况": "",
   "出院医嘱": "",
   "医师签名": "",
   "出院诊断": [{"诊断名称": "", "诊断标识": ""}, {"诊断名称": "", "诊断标识": ""}]
}

format_inpatient_map = {
   "format_admission_record": format_admission_record,
   "format_first_page": format_first_page,
   "format_progress_note": format_progress_note,
   "format_surgical_record": format_surgical_record,
   "format_informed_consent": format_informed_consent,
   "format_notification": format_notification,
   "format_discharge_summary": format_discharge_summary,
   "format_discharge_record": format_discharge_summary
}

inpatient_fields = {
   "姓名": "patient_name",
   "性别": "patient_gender",
   "年龄": "patient_age",
   "民族": "ethnicity",
   "职业": "occupation",
   "婚姻状况": "marital_status",
   "出生地": "birthplace",
   "入院时间": "admission_time",
   "记录时间": "recording_time",
   "病史陈述者": "history_reporter",
   "主诉": "chief_complaint",
   "现病史": "history_of_present_illness",
   "既往史": "past_medical_history",
   "体格检查": "physical_examination",
   "体温": "temperature",
   "脉搏": "pulse",
   "血压": "blood_pressure",
   "呼吸": "respiration",
   "辅助检查": "auxiliary_examination",
   "医师签名": "signature_of_physician",
   "初步诊断": "initial_diagnosis",
   "诊断名称": "diagnosis_name",
   "诊断标识": "diagnosis_identifier",

   "医疗机构名称": "medical_institution_name",
   "医疗机构代码": "medical_institution_code",
   "病案号": "medical_record_number",
   "身份证号": "identify_number",
   "入院日期": "admission_date",
   "出院日期": "discharge_date",
   "入院途径": "admission_route",
   "出院方式": "discharge_method",
   "主要诊断": "principal_diagnosis",
   "其他诊断": "other_diagnosis",
   "手术名称": "surgery_name",
   "手术日期": "surgery_date",
   "手术级别": "surgery_level",
   "切口愈合等级": "wound_healing_grade",
   "医疗付费方式": "medical_payment_method",
   "总费用": "total_expenses",
   "主治医师签名": "signature_of_attending_physician",
   "责任护士签名": "signature_of_responsible_nurse",
   "编码员签名": "signature_of_coder",
   "科主任签名": "signature_of_department_director",

   "记录日期": "recording_data",
   "病情变化": "change_in_condition",
   "检查结果": "examination_results",
   "诊疗措施": "therapeutic_measures",
   "医嘱调整理由": "reasons_for_medical_order_adjustment",
   "上级医师查房意见": "opinions_from_superior_physician_ward_round",
   "抢救记录": "rescue_record",

   "手术开始时间": "surgery_start_time",
   "手术结束时间": "surgery_end_time",
   "术者姓名": "name_of_surgeon",
   "助手姓名": "name_of_assistant_surgeon",
   "麻醉方式": "anesthesia_method",
   "麻醉医师姓名": "name_of_anesthesiologist",
   "手术步骤": "surgical_steps",
   "出血量": "blood_loss_volume",
   "输血量": "blood_transfusion_volume",
   "器械敷料清点": "inventory_of_instruments_and_dressings",
   "术后去向": "postoperative_disposition",
   "术中诊断": "intraoperative_diagnosis",

   "操作/手术名称": "name_of_operation",
   "预期获益": "expected_benefits",
   "可预见风险": "foreseeable_risks",
   "替代治疗方案": "alternative_treatment_plans",
   "风险应对措施": "risk_response_measures",
   "患者声明": "patient_statement",
   "患者签名": "signature_of_patient",
   "签署日期": "signing_date",

   "患者姓名": "patient_name",
   "病危/重依据": "basis_of_critical_condition",
   "预后风险": "prognostic_risks",
   "告知医师签名": "signature_of_informing_physician",
   "接收方签名": "signature_of_recipient",
   "关系证明": "proof_of_relationship",
   "签署时间": "signing_date",
   "诊断": "diagnosis",

   "住院号": "hospitalization_number",
   "住院天数": "length_of_hospital_stay",
   "入院主诉": "admission_chief_complaint",
   "重要检查结果": "important_examination_results",
   "手术/操作名称": "name_of_surgery",
   "出院情况": "discharge_condition",
   "出院医嘱": "discharge_instructions",
   "出院诊断": "discharge_diagnosis",
}

reversed_inpatient_fields = {v: k for k, v in inpatient_fields.items()}


#问候语
greetings_prompt="您好，请详细描述您的症状，主要说明哪里不舒服，持续了多久。可以参考以下案例来描述：\
\n<span style='color: blue'>“胃痉挛，胃部隐痛，上腹部疼痛，持续一天</span>”"

#每句后添加的
single_add_prompt="\n每次只提问一个问题。"
single_max_round="\n请生成病历。"
single_min_round="请问您还有其他补充的吗？"
first_round="您的就诊档案已经建立成功，欢迎您来我院就诊，期望我们可以帮助到您。"
irrelevant_content="\n如果我提到了与当前任务不相关的话题，必须给出不要讨论无关话题的提醒。"
#multi_agent_prompt="\n如果我有表达“建档、预问诊、挂号、缴费、报告查询”的意思，你需要直接返回【XXX链接】。\
#例如：【建档链接】、【预问诊链接】、【挂号链接】、【缴费链接】、【报告查询链接】。返回链接时先说“我为您找到了如下链接：”。"
multi_agent_prompt=""

stop_sign = [
    '现在为您返回',
    '已经为您生成了预问诊报告，如无问题，请点击确认',
    '如下预约就诊，您看是否可以？',
    '抱歉，目前没有查询到'
]

format_new_regiter_info = "新挂号"
format_register_first_info = "我们为您推荐如下预约就诊"

class PromptTemplate():
    def __init__(self) -> None:
        self.prompt = {}

    def set_prompt(self):
        return self.prompt

    def get_prompt(self, flag):
        return self.prompt.get(str(flag))

def register_prompt(cls):
    scene, ver = cls.__name__.split("_")
    VersionedPromptFactory[scene][ver] = cls
    return cls

def get_prompt(scene: str, req, *args):
    ver = req.chat.prompt_version
    if ver == "":
        ver = max(VersionedPromptFactory[scene])
        req.chat.prompt_version = ver
    if ver not in VersionedPromptFactory[scene]:
        print(f"Error not found version {ver} fallback to v1", flush=True)
        raise NotImplemented(f"not implement {ver} version in {scene} prompt")
    return VersionedPromptFactory[scene][ver](req, *args)

class PromptManager:
    def __init__(self, yaml_name):
        try:
            self.yaml_name = yaml_name
            prompt_config_path = "./diagnosis_treatment/prompt_config"
            yaml_path = os.path.join(os.getcwd(), prompt_config_path, yaml_name)
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.data = yaml.safe_load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"YAML file '{yaml_name}' not found in path '{prompt_config_path}'")
        except yaml.YAMLError as e:
            raise HTTPException(status_code=422, detail=f"Error parsing YAML file: {yaml_name}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error loading YAML: {e}")

    def get_prompt(self, category, index, variables):
        try:
            template_str = self.data["prompts"][category][index]['prompt']
            template = Template(template_str)
            return template.render(**variables)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error getting prompt: {e}")
