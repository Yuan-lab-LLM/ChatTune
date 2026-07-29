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

import gradio as gr
from .tabs import *
from ..util import inference_gradio_json_data
from ..config import prompt_versions, REGISTER_INTENTION_TYPE
from ..logic.handlers import *
from frontend.ui.quality_tab import create_quality_inspect_interface, create_quality_modify_interface

def create_interface_hospitalguide():
    with gr.Blocks(analytics_enabled=False) as interface_hospitalguide:
        build_chat_tab(
            module_name="hospitalguide",
            json_data=inference_gradio_json_data['hospitalguide_simple'],
            prompt_name=prompt_versions["hospitalguide"],
            default_chat=None,
            function=hospitalguide,
            note="simple为“简单对话”生成主诉与推荐科室。<br>detailed根据已有主诉，“详细对话”生成预问诊报告。",
            use_branch=True,
            branch_content={"label":"HospitalGuide Type", "choices": ["simple", "detailed"], "value": "simple"}
        )
    return interface_hospitalguide

def create_interface_returnvisit():
    with gr.Blocks(analytics_enabled=False) as interface_returnvisit:
        json_display, module = build_chat_tab(
            module_name="returnvisit",
            #json_data={},
            json_data=inference_gradio_json_data['returnvisit'],
            prompt_name=prompt_versions["returnvisit"],
            default_chat=None,
            function=returnvisit,
            note="请先点击“疾病诊断”页面“生成诊断”，再点击“发送到 患者复诊”。"
        )
    return interface_returnvisit, json_display, module

def create_interface_hospitalregister():
    with gr.Blocks(analytics_enabled=False) as interface_hospitalregister:
        build_chat_tab(
            module_name="hospitalregister",
            json_data=inference_gradio_json_data['hospitalregister'],
            prompt_name=prompt_versions["hospitalregister"],
            default_chat=[[None, inference_gradio_json_data['hospitalregister']['chat']['historical_conversations'][-2]['content']], [None, inference_gradio_json_data['hospitalregister']['chat']['historical_conversations'][-1]['content']]],
            function=hospitalregister,
            use_branch=True,
            branch_content={"label":"Register Intention Tpye", "choices": REGISTER_INTENTION_TYPE, "value": max(REGISTER_INTENTION_TYPE)}
        )
    return interface_hospitalregister

def create_interface_basicmedicalrecord(json_display_diagnosis, diagnosis):
    with gr.Blocks(analytics_enabled=False) as interface_basicmedicalrecord:
        json_display, module = build_chat_tab(
            module_name="basicmedicalrecord",
            json_data=inference_gradio_json_data['basicmedicalrecord'],
            prompt_name=prompt_versions["basicmedicalrecord"],
            default_chat=[[None, inference_gradio_json_data['basicmedicalrecord']['chat']['historical_conversations'][-1]['content']]],
            function=basicmedicalrecord
        )
        build_send_button(
            btn_config=[
                {"label": "发送到 疾病诊断", "display": json_display_diagnosis, "module": diagnosis, "visible": True},
            ],
            result_json=json_display
        )
    return interface_basicmedicalrecord

def create_interface_clientinfo():
    with gr.Blocks(analytics_enabled=False) as interface_clientinfo:
        build_chat_tab(
            module_name="clientinfo",
            json_data=inference_gradio_json_data['clientinfo'],
            prompt_name=prompt_versions["clientinfo"],
            default_chat=[[None, inference_gradio_json_data['clientinfo']['chat']['historical_conversations'][-1]['content']]],
            function=clientinfo
        )
    return interface_clientinfo

def create_interface_distribute():
    with gr.Blocks(analytics_enabled=False) as interface_distribute:
        build_chat_tab(
            module_name="distribute",
            json_data=inference_gradio_json_data['distribute'],
            prompt_name=prompt_versions["distribute"],
            default_chat=[[None, inference_gradio_json_data['distribute']['chat']['historical_conversations'][-1]['content']]],
            function=distribute
        )
    return interface_distribute

def create_interface_doctormedicalrecord():
    with gr.Blocks(analytics_enabled=False) as interface_doctormedicalrecord:
        build_nochat_tab(
            module_name="doctormedicalrecord",
            json_data=inference_gradio_json_data['doctormedicalrecord_general'],
            prompt_name=prompt_versions["doctormedicalrecord"],
            module_label="病历",
            btn_name="生成病历",
            function=fetch_response_nochat,
            use_branch=True,
            branch_content={"label":"MedicalRecord Type", "choices": ["general", "special", "special_modify", "special_select"], "value": "general"}
        )
    return interface_doctormedicalrecord

def create_interface_inpatient():
    with gr.Blocks(analytics_enabled=False) as interface_inpatient:
        build_nochat_tab(
            module_name="inpatient",
            json_data=inference_gradio_json_data['inpatient']['admission_record'],
            prompt_name=prompt_versions["inpatient"],
            module_label="住院文书",
            btn_name="生成文书",
            function=fetch_response_nochat,
            use_branch=True,
            branch_content={
                "label":"Inpatient Type",
                "value": "admission_record",
                "choices": ["admission_record", "first_page", "progress_note", "surgical_record",
                    "informed_consent", "notification", "discharge_summary", "discharge_record"]
            }
        )
    return interface_inpatient

def create_interface_examass():
    with gr.Blocks(analytics_enabled=False) as interface_examass:
        json_display, module, *_ = build_nochat_tab(
            module_name="examass",
            json_data=inference_gradio_json_data['examass'],
            prompt_name=prompt_versions["examass"],
            module_label="检查与化验",
            btn_name="生成检查与化验",
            function=fetch_response_nochat,
            note="括号中内容为匹配数据表后的检查名称与化验名称。<br>请先点击“疾病诊断”页面“生成诊断”，再点击“发送到 检查化验开具”。",
        )
    return interface_examass, json_display, module

def create_interface_diagnosis(json_display_examass, examass, json_display_scheme, scheme, json_display_returnvisit, returnvisit):
    with gr.Blocks(analytics_enabled=False) as interface_diagnosis:
        json_display, module, result_json, send_btn, json_file, json_md, result_text, branch = build_nochat_tab(
            module_name="diagnosis",
            json_data=inference_gradio_json_data['diagnosis'],
            prompt_name=prompt_versions["diagnosis"],
            module_label="诊断",
            btn_name="生成诊断",
            function=fetch_response_nochat,
            note="括号中内容为匹配数据表后的诊断名称。"
        )
        extra_btns = build_send_button(
            btn_config=[
                {"label": "发送到 检查化验", "display": json_display_examass, "module": examass, "visible": False},
                {"label": "发送到 治疗方案", "display": json_display_scheme, "module": scheme, "visible": False},
                {"label": "发送到 患者复诊", "display": json_display_returnvisit, "module": returnvisit, "visible": False}
            ],
            result_json=result_json
        )
        #send_btn.click(fetch_response_nochat,
        #    inputs=[json_display, json_file, module],
        #    outputs=[json_file, json_md, result_text, result_json, *extra_btns]
        #)
        send_btn.click(fetch_response_nochat,
            inputs=[json_display, json_file, module, json_md, result_text, result_json, branch],
            outputs=[json_display, json_file, module, json_md, result_text, result_json, branch, *extra_btns]
        )
    return interface_diagnosis, json_display, module

def create_interface_scheme():
    with gr.Blocks(analytics_enabled=False) as interface_scheme:
        with gr.TabItem("📑️ Step 1: 挑选方案"):
            json_display, module, result_json, send_btn, json_file, json_md, result_text, *_ = build_nochat_tab(
                module_name="scheme",
                json_data=inference_gradio_json_data['scheme'],
                prompt_name=prompt_versions["scheme"],
                module_label="多方案",
                btn_name="生成多方案",
                function=fetch_response_pick_scheme,
                note="请先点击“疾病诊断”页面“生成诊断”，再点击“发送到 治疗方案”。"
            )
        with gr.TabItem("️💊 Step 2: 生成方案"):
            therapy, *_, therapy_branch = build_nochat_tab(
                module_name="therapy",
                json_data={},
                prompt_name=prompt_versions["scheme"],
                module_label="治疗方案",
                btn_name="生成 治疗方案",
                function=fetch_response_generate_therapy,
                note="请先点击“Step 1: 挑选方案”页面“生成多方案”。",
                use_branch=True,
                branch_content={"label":"Therapy Id", "choices": ["1"], "value": "1"}
            )
        with gr.TabItem("️✨ Test: 批量测试"):
            build_nochat_tab(
                module_name="scheme_test",
                json_data=inference_gradio_json_data['scheme'],
                prompt_name=prompt_versions["scheme"],
                module_label="多方案",
                btn_name="生成多方案",
                function=fetch_response_generate_all_therapy
            )

        send_btn.click(fetch_response_pick_scheme,
            inputs=[json_display, json_file, module],
            outputs=[json_file, json_md, result_text, result_json,
                therapy, therapy_branch
            ]
        )

    return interface_scheme, json_display, module

interface_examass, json_display_examass, examass = create_interface_examass()
interface_scheme, json_display_scheme, scheme = create_interface_scheme()
interface_returnvisit, json_display_returnvisit, return_visit = create_interface_returnvisit()
interface_diagnosis, json_display_diagnosis, diagnosis = create_interface_diagnosis(
    json_display_examass, examass, json_display_scheme, scheme, json_display_returnvisit, return_visit
)

interface_distribute = create_interface_distribute()
interface_clientinfo = create_interface_clientinfo()
interface_basicmedicalrecord = create_interface_basicmedicalrecord(json_display_diagnosis, diagnosis)
interface_hospitalregister = create_interface_hospitalregister()
interface_inpatient = create_interface_inpatient()

interface_hospitalguide = create_interface_hospitalguide()
interface_doctormedicalrecord = create_interface_doctormedicalrecord()

quality_modify_interface, json_display_quality_modify = create_quality_modify_interface()
quality_inspect_interface = create_quality_inspect_interface(json_display_quality_modify)