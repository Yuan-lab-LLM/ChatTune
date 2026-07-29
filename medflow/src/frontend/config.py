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

import argparse

prompt_versions = {
    "distribute": ["v2"],
    "clientinfo": ["v5"],
    "basicmedicalrecord": ["v5"],
    "hospitalregister": ["v6"],
    "diagnosis": ["v2"],
    "examass": ["v2"],
    "scheme": ["v5"],
    "returnvisit": ["v2"],
    "hospitalguide": ["v4"],
    "doctormedicalrecord": ["v3"],
    "inpatient": ["v1"],
    "quality_inspect": ["v2"],
    "quality_modify": ["v2"]
}

voice_model = {
    "asr": [
        "whisper-large-v3",
        "dolphin-small",
    ],
    "tts": [
        "CosyVoice-300M-SFT",
        "Bert-VITS2",
        "Gpt-SoVITS",
    ]
}

voice_config = {
    "doctormedicalrecord":[
        ["./frontend/voice/I期输入.mp3", "{\"input\": {\"medical_templet\": \"主诉：复诊，无不适，行种植手术。现病史：已完善种植相关检査，要求种植。既往史：同前。专科检查：同前。治疗:（牙编号）碧兰麻局部浸润麻醉，牙龈切开，翻瓣，牙槽骨修整。定位，逐级进行窝洞预备。植入（品牌及尺寸）种植体（数值）枚，接（）。（）填入（），覆盖（）。牙龈修整成型，（间断、褥式、褥式+间断）缝合。（牙编号）种植体植入情况:牙龈厚度:（数值）mm。种植系统选择：（品牌及尺寸）。植入时的扭力:（数值）N.cm。愈合方式:（）。\",\"templet_type\": \"1\",\"doctor_supplement\": \"常规一期手术，36植入种植体Astra EV 4.2mm*11mm种植体1枚，接覆盖螺丝，植入bio-oss骨粉，唇侧植入，覆盖bio-gide胶原膜，牙龈修整成型，褥式+间断缝合。36牙龈厚度2mm，植入扭矩35N，埋入愈合。\"}}", "special_select"],
        ["./frontend/voice/II期输入.mp3", "{\"input\": {\"medical_templet\": \"主诉：复诊行种植二期手术。现病史：患者于（）已完成种植一期手术，术后无不适，现复诊进行种植二期手术。既往史：同前。专科检查：（牙编号）缺牙区牙龈（），角化龈宽度（），呈（）辅助检查：X片显示（） 治疗：行（牙编号）种植二期手术。常规消毒铺巾，（牙编号）碧兰麻局部浸润麻醉，牙龈切开、翻瓣，取出封闭螺丝，安装直径（数值）mm、高（数值）mm愈合基台，（）。常嘱，不适随诊。\",\"templet_type\": \"1\",\"doctor_supplement\": \"患者两月前完成了种植一期手术，37牙龈无红肿现象，角化龈宽度尚可，呈薄龈生物型，根尖片显示37种植体周围无阴影。在37种植二期手术中，安装了6*6mm的愈合基台，并进行了缝合。\"}}", "special_select"],
        ["./frontend/voice/取模输入.mp3", "{\"input\": {\"medical_templet\": \"主诉：种植二期手术后（时间跨度）复诊取模。现病史：患者（时间跨度）已完成种植（）手术，（），现复诊进行种植取模。既往史：同前专科检查：（牙编号）缺牙区愈合基台（），牙龈（）。邻牙（），对颌牙（），修复空间（）。辅助检查：X片显示（）治疗：（牙编号）卸下愈合基台，连接（），（），冲洗，安装愈合基台，比色（）。\",\"templet_type\": \"1\",\"doctor_supplement\": \"患者于一周前完成了种植二期手术，术后无不适。缺牙区的愈合基台在位，牙龈无红肿现象，邻近牙齿无龋损或倾斜，对合牙齿无伸长，修复空间尚可。根尖片检查34种植体周围无阴影。进行了34连接数字化转移杆的操作，并通过口扫取了上下颌的印模及咬合关系，进行了比色3M2。\"}}", "special_select"],
        ["./frontend/voice/戴牙输入.mp3", "{\"input\": {\"medical_templet\": \"主诉：复诊，无不适，戴牙。现病史：患者（时间跨度）前于我科行种植取模，无不适，今日复诊戴牙。既往史：同前。专科检查：（牙编号）缺牙区愈合基台（），（），牙龈（），全口口腔卫生（）。治疗：（牙编号）义齿试戴，就位良好，边缘密合，邻面接触良好。口外粘接义齿，戴入后（数值）N.cm锁紧。X片显示：义齿边缘密合，邻接关系可，无粘结剂残留。树脂封螺丝孔，调牙合抛光。\",\"templet_type\": \"1\",\"doctor_supplement\": \"患者在3周前完成了种植牙的取模，27位点的愈合基台在位且没有松动，牙龈也没有红肿，口腔卫生状况良好。27位点的种植牙边缘密合，就位良好，邻接关系正常，戴入后锁紧力度为22N。\"}}", "special_select"]
    ],
    "inpatient":[
        ["./frontend/voice/入院记录.mp3", "{\"input\": {\"doctor_supplement\": \"姓名张三，性别男，49岁，汉族，已婚，广东省广州市天河区珠江新城12号院，2025年6月3号入院的，记录时间是19号下午3点，陈述者是李斯，副主任医师。主诉解黑便11小时。现病史昨晚20时开始无明显诱因出现解黑便2次，呈黑色成形大便，每次量约100g，伴阵发性腹部隐痛，偶有头晕、出冷汗，有气促，无晕厥，无胸痛、胸闷，无呕吐，无发热。既往史胃窦部溃疡伴出血、肺部感染、糜烂性胃炎、十二指肠炎、贫血、高血压病和腰椎退行性病变，骨质疏松，电解质紊乱，老年性心脏瓣膜病，高血压病史多年，长期服药治疗，否认糖尿病。体温36.5℃，脉搏86次/分，血压150/61mmHg，呼吸18次/分。诊断为上消化道出血和贫血。\"}}", "admission_record"],
        ["./frontend/voice/病案首页.mp3", "{\"input\": {\"doctor_supplement\": \"医疗机构人民医院，机构代码是NHRY7895245，张三，女，54岁，BL156325420，身份证102312023632063655，入院时间2026年6月23日，出院6月29日，入院途径是门诊，出院方式好转，主要诊断消化道出血，其他诊断是贫血，手术名称上消化道止血术，日期6月25日，手术级别4，I类伤口愈合，城镇职工医保付费，总费用35236.78元，主治医师李斯，责任护士王武，编码员赵柳，科室主任孙琪\"}}", "first_page"],
        ["./frontend/voice/病程记录.mp3", "{\"input\": {\"doctor_supplement\": \"2025年6月25日记录，病情体温升高，胃痛严重，检查结果显示胃出血，准备服用止血药物，后续严重将进行手术，如有需要进行抢救，张三医生\"}}", "progress_note"],
        ["./frontend/voice/手术记录.mp3", "{\"input\": {\"doctor_supplement\": \"上消化道止血术，2025年6月25日，上午8:30开始，10:36结束，张三主刀，助理李思，全麻，麻醉是王武，术中确认贫血，出血500ml，输血红细胞4U,清点无误，术后返回病房，张三\"}}", "surgical_record"],
        ["./frontend/voice/知情同意书.mp3", "{\"input\": {\"doctor_supplement\": \"腹腔镜胆囊切除术，延长生存期，有出血、感染、脏器损伤风险，可替代方案为保守治疗，应急处理二次手术，患者自愿参与，患者张三，医师李四，2025年6月26日\"}}", "informed_consent"],
        ["./frontend/voice/通知书.mp3", "{\"input\": {\"doctor_supplement\": \"患者张三，诊断上消化道出血，贫血，病危，多器官衰竭，预后风险为植物状态，医师李思，家属王武，父子，2025年6月26日签署\"}}", "notification"],
        ["./frontend/voice/出院小结.mp3", "{\"input\": {\"doctor_supplement\": \"患者张三，性别男，年龄74，住院号ZY45871256，于2025年6月25日入院，7月3号出院，实际住院9天，主诉解黑便1周，检查结果显示胃出血，进行上消化道止血手术，最终诊断上消化道出血，症状改善后出院，嘱咐患者术后饮食清淡，流质食物缓慢过度，于1个月后复诊，主治医师王武\"}}", "discharge_summary"],
        ["./frontend/voice/出院记录.mp3", "{\"input\": {\"doctor_supplement\": \"患者张三，性别男，年龄74，住院号ZY45871256，于2025年6月25日入院，7月3号出院，实际住院9天，主诉解黑便1周，检查结果显示胃出血，进行上消化道止血手术，最终诊断上消化道出血，症状改善后出院，嘱咐患者术后饮食清淡，流质食物缓慢过度，于1个月后复诊，主治医师王武\"}}", "discharge_record"]
    ]
}

REGISTER_INTENTION_TYPE = ["base_tpye", "intention_tpye"]

inference_gradio_http_common_headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

def args_parser():
    parser = argparse.ArgumentParser(description='Chatbot Web Interface with Customizable Parameters')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8015")
    parser.add_argument("--gradio-port", type=int, default="7015")
    parser.add_argument("--share", action="store_true", help="Whether to generate a public, shareable link")
    parser.add_argument("--concurrency-count", type=int, default=50, help="The concurrency count of the gradio queue")
    parser.add_argument('--model', type=str, required=True, help='Model name for the chatbot')
    parser.add_argument('--voice-url', type=str, default='http://localhost:9997/v1', help='Voice Model URL')
    args = parser.parse_args()
    return args

args = args_parser()
