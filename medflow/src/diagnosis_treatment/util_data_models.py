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

from typing import Generic, List, TypeVar, Union

from pydantic import BaseModel, Field, model_validator

Inpatient = TypeVar("Inpatient")

class HistoricalConversations(BaseModel):
    role: str = ""
    content: str = ""

class Chat(BaseModel):
    historical_conversations_bak: List[HistoricalConversations] = Field(default_factory=list)
    historical_conversations: List[HistoricalConversations] = Field(default_factory=list)
    prompt_version: str = ""
    model_name: str = ""

class PhyscialExamination(BaseModel):
    temperature: str = ""
    pulse: str = ""
    blood_pressure: str = ""
    respiration: str = ""

class BasicMedicalRecord(BaseModel):
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: str
    personal_history: str
    allergy_history: str
    physical_examination: PhyscialExamination
    auxiliary_examination: str

class DoctorMedicalRecord(BaseModel):
    chief_complaint: str = ""
    history_of_present_illness: str = ""
    past_medical_history: str = ""
    personal_history: str = ""
    allergy_history: str = ""
    physical_examination: PhyscialExamination = Field(default_factory=PhyscialExamination)
    auxiliary_examination: str = ""
    specialty_examination: str = ""
    cure: str = ""
    doctor_advice: str = ""

# v0  zh: fen fa, distribute
class PatientNeed(BaseModel):
    need: str

class OutputV0(BaseModel):
    patient_need: List[PatientNeed]

class RequestV0(BaseModel):
    input: None
    output: OutputV0
    chat: Chat


# v1  zh: jian dang, clinet info
class CurrentAddress(BaseModel):
    province: str
    city: str
    district: str
    street: str

class PatientBasic(BaseModel):
    patient_name: str
    patient_gender: str
    patient_age: str

class Patient(PatientBasic):
    if_child: str
    certificate_type: str
    certificate_number: str
    mobile_number: str
    current_address: CurrentAddress
    detailed_address: str

class Guardian(BaseModel):
    guardian_name: str
    certificate_type: str
    certificate_number: str

class ClientInfo(BaseModel):
    patient: Patient
    guardian: Guardian

class InputV1(BaseModel):
    client_info: Union[List[ClientInfo], List[None]]

class OutputV1(BaseModel):
    client_info: List[ClientInfo]
    create_file: str

class RequestV1(BaseModel):
    input: InputV1
    output: OutputV1
    chat: Chat


# v2  zh: yu wen zhen, basic medical record
class InputV2(BaseModel):
    client_info: Union[List[ClientInfo], List[None]]

class OutputV2(BaseModel):
    basic_medical_record: BasicMedicalRecord

class RequestV2(BaseModel):
    input: InputV2
    output: OutputV2
    chat: Chat


# v3  zh: gua hao, register diagnosis
class RegisterIntention(BaseModel):
    intention_tpye: str | None = None
    department_name: str| None = None
    doctor_name: str | None = None
    doctor_title: str | None = None
    register_date: str | None = None
    register_time: str | None = None
    register_source: str | None = None
    query_subject: str | None = None
    priority: str | None = None

class TimeList(BaseModel):
    start_time: str
    end_time: str
    source_num: str

class DateList(BaseModel):
    date: str
    time_list: List[TimeList]

class DoctorList(BaseModel):
    doctor_id: str
    doctor_name: str
    doctor_title: str
    date_list: List[DateList] | None = None

class Department(BaseModel):
    department_id: str
    department_name: str

class HospitalRegister(Department):
    doctor_list: List[DoctorList]    
    
class InputV3(BaseModel):
    client_info: List[ClientInfo] | None = None
    basic_medical_record: BasicMedicalRecord
    all_department: List[Department] | None = None
    hospital_register: Union[List[HospitalRegister], List[None]] | None = None
    register_intention_enable: bool | None = None

class OutputV3(BaseModel):
    chosen_department: List[Department]
    hospital_register: List[HospitalRegister]
    register_intention_enable: bool | None = None
    register_intention_info: List[RegisterIntention]  | None = None

class RequestV3(BaseModel):
    input: InputV3
    output: OutputV3
    chat: Chat


# v4  zh: zhen duan, diagnosis
class Diagnosis(BaseModel):
    diagnosis_name: str = ""
    diagnosis_name_retrieve: str = ""
    diagnosis_code: str = ""
    diagnosis_identifier: str = ""

class InputV4(BaseModel):
    client_info: Union[List[ClientInfo], List[None]]
    basic_medical_record: BasicMedicalRecord

class OutputV4(BaseModel):
    diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class RequestV4(BaseModel):
    input: InputV4
    output: OutputV4 = Field(default_factory=OutputV4)
    chat: Chat = Field(default_factory=Chat)


# v5  zh: jian cha hua yan, examine assay
class InputV5(BaseModel):
    client_info: Union[List[ClientInfo], List[None]]
    basic_medical_record: BasicMedicalRecord
    diagnosis: List[Diagnosis]

class ExamineContent(BaseModel):
    examine_code: str = ""
    examine_category: str = ""
    examine_name: str = ""
    examine_name_retrieve: str = ""
    order_quantity: str = ""
    examine_result: str = ""
    corresponding_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class AssayResult(BaseModel):
    result_id: str = ""
    result_item: str = ""
    result_value: float = 0.0
    result_identifier: str = ""
    reference_value: str = ""
    unit: str = ""
    
class AssayContent(BaseModel):
    assay_code: str = ""
    assay_category: str = ""
    assay_name: str = ""
    assay_name_retrieve: str = ""
    order_quantity: str = ""
    assay_result: List[AssayResult] = Field(default_factory=lambda: [AssayResult()])
    corresponding_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class OutputV5(BaseModel):
    examine_content: List[ExamineContent] = Field(default_factory=lambda: [ExamineContent()])
    assay_content: List[AssayContent] = Field(default_factory=lambda: [AssayContent()])

class RequestV5(BaseModel):
    input: InputV5
    output: OutputV5 = Field(default_factory=OutputV5)
    chat: Chat = Field(default_factory=Chat)


# v6  zh: zhi liao fang an, therapy scheme
class TherapyFields(BaseModel):
    field_id: str
    field_name: str

class InputV6(BaseModel):
    client_info: Union[List[ClientInfo], List[None]]
    basic_medical_record: BasicMedicalRecord
    diagnosis: List[Diagnosis]
    therapy_fields: List[TherapyFields]

class PrescriptionContent(BaseModel):
    drug_id: str
    drug_name: str
    drug_name_retrieve: str
    drug_specification: str
    manufacturer_name: str
    order_quantity: str
    order_unit: str
    route_of_administration: str
    dosage: str
    duration: str
    frequency: str
    corresponding_diseases: str
    drug_efficacy: str

class Prescription(BaseModel):
    prescription_content: List[PrescriptionContent]

  
class TransfusionContent(PrescriptionContent):
    infusion_group: str
    infusion_rate: str
        
class Transfusion(BaseModel):
    transfusion_content: List[TransfusionContent]


class DispositionContent(BaseModel):
    disposition_id: str
    disposition_name: str
    disposition_name_retrieve: str
    frequency: str
    dosage: str
    duration: str
    
class Disposition(BaseModel):
    disposition_content: List[DispositionContent]


class TherapyInterpret(BaseModel):
    therapy_name: str
    therapy_content: str

class PickTherapy(BaseModel):
    therapy_name: str
    therapy_summary: str
    therapy_interpret: List[TherapyInterpret]


class MethodTherapyContent(BaseModel):
    method_code: str
    method_type: str
    method_name: str
    method_name_retrieve: str
    corresponding_diseases: str
    method_plan: str
    method_risk: str

class TherapyContent(BaseModel):
    prescription: List[PrescriptionContent] = Field(default_factory=list)
    transfusion: List[TransfusionContent] = Field(default_factory=list)
    disposition: List[DispositionContent] = Field(default_factory=list)
    examine: List[ExamineContent] = Field(default_factory=list)
    assay: List[AssayContent] = Field(default_factory=list)
    surgical: List[MethodTherapyContent] = Field(default_factory=list)
    chemo: List[MethodTherapyContent] = Field(default_factory=list)
    radiation: List[MethodTherapyContent] = Field(default_factory=list)
    psycho: List[MethodTherapyContent] = Field(default_factory=list)
    rehabilitation: List[MethodTherapyContent] = Field(default_factory=list)
    physical: List[MethodTherapyContent] = Field(default_factory=list)
    alternative: List[MethodTherapyContent] = Field(default_factory=list)
    observation: List[MethodTherapyContent] = Field(default_factory=list)

class GenerateTherapy(BaseModel):
    therapy_name: str = ""
    therapy_content: TherapyContent = Field(default_factory=lambda: TherapyContent())

class OutputV6(BaseModel):
    pick_therapy: List[PickTherapy] = Field(default_factory=list)
    generate_therapy: List[GenerateTherapy] = Field(default_factory=lambda: [GenerateTherapy()])

class RequestV6(BaseModel):
    input: InputV6
    output: OutputV6 = Field(default_factory=OutputV6)
    chat: Chat = Field(default_factory=Chat)


# v7  zh: fu zhen, return visit
class ReturnVisit(BaseModel):
    summary: str
    if_visit: str

class InputV7(BaseModel):
    client_info: List[ClientInfo]
    basic_medical_record: BasicMedicalRecord
    diagnosis: List[Diagnosis]

class OutputV7(BaseModel):
    return_visit: ReturnVisit

class RequestV7(BaseModel):
    input: InputV7
    output: OutputV7
    chat: Chat


# v8  zh: dao zhen, hospital guide
class Department2(BaseModel):
    department_id: str
    department_name: str
    if_child: bool

class InputV8(BaseModel):
    client_info: List[ClientInfo]
    all_department: List[Department2] | None = None

class OutputV8(BaseModel):
    basic_medical_record: BasicMedicalRecord
    chosen_department: List[Department2]

class RequestV8(BaseModel):
    input: InputV8
    output: OutputV8
    chat: Chat


# v9  zh: bing li, doctor medical record
class InputV9(BaseModel):
    medical_templet: str | None = None
    templet_type: str | None = None
    basic_medical_record: DoctorMedicalRecord = Field(default_factory=DoctorMedicalRecord)
    doctor_supplement: str

class OutputV9(BaseModel):
    medical_format: str | None = None
    basic_medical_record: DoctorMedicalRecord | None = None

class RequestV9(BaseModel):
    input: InputV9
    output: OutputV9 = Field(default_factory=OutputV9)
    chat: Chat  = Field(default_factory=Chat)


# inpatient
class AdmissionRecord(BaseModel):
    patient_name: str = ""
    patient_gender: str = ""
    patient_age: str = ""
    ethnicity: str = ""
    occupation: str = ""
    marital_status: str = ""
    birthplace: str = ""
    admission_time: str = ""
    recording_time: str = ""
    history_reporter: str = ""
    chief_complaint: str = ""
    history_of_present_illness: str = ""
    past_medical_history: str = ""
    physical_examination: PhyscialExamination = Field(default_factory=PhyscialExamination)
    auxiliary_examination: str = ""
    signature_of_physician: str = ""
    initial_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class FirstPage(BaseModel):
    medical_institution_name: str = ""
    medical_institution_code: str = ""
    patient_name: str = ""
    patient_gender: str = ""
    patient_age: str = ""
    medical_record_number: str = ""
    identify_number: str = ""
    admission_date: str = ""
    discharge_date: str = ""
    admission_route: str = ""
    discharge_method: str = ""
    surgery_name: str = ""
    surgery_date: str = ""
    surgery_level: str = ""
    wound_healing_grade: str = ""
    medical_payment_method: str = ""
    total_expenses: str = ""
    signature_of_attending_physician: str = ""
    signature_of_responsible_nurse: str = ""
    signature_of_coder: str = ""
    signature_of_department_director: str = ""
    principal_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])
    other_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class ProgressNote(BaseModel):
    recording_data: str = ""
    change_in_condition: str = ""
    examination_results: str = ""
    therapeutic_measures: str = ""
    reasons_for_medical_order_adjustment: str = ""
    opinions_from_superior_physician_ward_round: str = ""
    rescue_record: str = ""
    signature_of_physician: str = ""

class SurgicalRecord(BaseModel):
    surgery_name: str = ""
    surgery_date: str = ""
    surgery_start_time: str = ""
    surgery_end_time: str = ""
    name_of_surgeon: str = ""
    name_of_assistant_surgeon: str = ""
    anesthesia_method: str = ""
    name_of_anesthesiologist: str = ""
    surgical_steps: str = ""
    blood_loss_volume: str = ""
    blood_transfusion_volume: str = ""
    inventory_of_instruments_and_dressings: str = ""
    postoperative_disposition: str = ""
    signature_of_physician: str = ""
    intraoperative_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class InformedConsentForm(BaseModel):
    name_of_operation: str = ""
    expected_benefits: str = ""
    foreseeable_risks: str = ""
    alternative_treatment_plans: str = ""
    risk_response_measures: str = ""
    patient_statement: str = ""
    signature_of_patient: str = ""
    signature_of_physician: str = ""
    signing_date: str = ""

class Notification(BaseModel):
    patient_name: str = ""
    basis_of_critical_condition: str = ""
    prognostic_risks: str = ""
    signature_of_informing_physician: str = ""
    signature_of_recipient: str = ""
    proof_of_relationship: str = ""
    signing_date: str = ""
    diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class Discharge(BaseModel):
    patient_name: str = ""
    patient_gender: str = ""
    patient_age: str = ""
    hospitalization_number: str = ""
    admission_date: str = ""
    discharge_date: str = ""
    length_of_hospital_stay: str = ""
    admission_chief_complaint: str = ""
    important_examination_results: str = ""
    name_of_surgery: str = ""
    discharge_condition: str = ""
    discharge_instructions: str = ""
    signature_of_physician: str = ""
    discharge_diagnosis: List[Diagnosis] = Field(default_factory=lambda: [Diagnosis()])

class InputInpatient(BaseModel, Generic[Inpatient]):
    doctor_supplement: str
    included_fields: list[str] | None = None
    inpatient: Inpatient
    @model_validator(mode="after")
    def set_default(self):
        if self.included_fields is None:
            self.included_fields = list(self.inpatient.__class__.model_fields.keys())
        return self

class OutputInpatient(BaseModel, Generic[Inpatient]):
    inpatient: Inpatient

class RequestInpatient(BaseModel, Generic[Inpatient]):
    input: InputInpatient[Inpatient]
    output: OutputInpatient[Inpatient]
    chat: Chat = Field(default_factory=Chat)

def build_input(Inpatient):
    class Input(InputInpatient[Inpatient]):
        inpatient: Inpatient = Field(default_factory=Inpatient)
    return Input

def build_output(Inpatient):
    class Output(OutputInpatient[Inpatient]):
        inpatient: Inpatient = Field(default_factory=Inpatient)
    return Output

def build_request(Inpatient):
    Input = build_input(Inpatient)
    Output = build_output(Inpatient)
    class Request(RequestInpatient[Inpatient]):
        input: Input
        output: Output = Field(default_factory=Output)
    return Request

RequestAdmissionRecord = build_request(AdmissionRecord)
RequestFirstPage = build_request(FirstPage)
RequestProgressNote = build_request(ProgressNote)
RequestSurgicalRecord = build_request(SurgicalRecord)
RequestInformedConsentForm = build_request(InformedConsentForm)
RequestNotification = build_request(Notification)
RequestDischarge = build_request(Discharge)
