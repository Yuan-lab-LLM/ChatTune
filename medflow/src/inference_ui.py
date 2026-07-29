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
from frontend.config import args
from frontend.ui.ui import (
    interface_distribute,
    interface_clientinfo,
    interface_basicmedicalrecord,
    interface_hospitalregister,
    interface_diagnosis,
    interface_examass,
    interface_scheme,
    interface_returnvisit,
    interface_hospitalguide,
    interface_doctormedicalrecord,
    interface_inpatient,
    quality_inspect_interface,
    quality_modify_interface
)

block_css = """
#shared_group div.svelte-1nguped {
    background: white;
}
"""

def build_app():
    with gr.Blocks(title="MedFlow", theme="soft", css=block_css) as app:
        with gr.Row():
            gr.Markdown("## **🏥 MedFlow**")
        with gr.Row():
            with gr.Column(scale=1, min_width="600px"):
                state = gr.State(value = [False, False, False])
                pre_btn = gr.Button("ℹ️ 诊前模块", variant="primary", size="md")
                distribute_btn = gr.Button("🏷️ 任务分发", size="md", visible=False)
                clientinfo_btn = gr.Button("👤 患者建档", size="md", visible=False)
                basicmedicalrecord_btn = gr.Button("🗣️ 症状预问诊", size="md", visible=False)
                hospitalguide_btn = gr.Button("🪧 导诊推荐科室", size="md", visible=False)
                hospitalregister_btn = gr.Button("🏢 智能挂号", size="md", visible=False)

                in_btn = gr.Button("🥼 诊中模块", variant="primary", size="md")
                quality_inspect_btn = gr.Button("🔍 病历质检-检验", size="md", visible=False)
                quality_modify_btn = gr.Button("✍️ 病历质检-修改", size="md", visible=False)
                doctormedicalrecord_btn = gr.Button("📄 病历生成", size="md", visible=False)
                diagnosis_btn = gr.Button("🧬 疾病诊断", size="md", visible=False)
                examass_btn = gr.Button("🧪 检查化验开具", size="md", visible=False)
                scheme_btn = gr.Button("🌿 治疗方案", size="md", visible=False)
                inpatient_btn = gr.Button("🛏️ 住院文书", size="md", visible=False)

                post_btn = gr.Button("📋️ 诊后模块", variant="primary", size="md")
                returnvisit_btn = gr.Button("🔁 患者复诊", size="md", visible=False)

            with gr.Column(scale=15):
                with gr.Group(visible=False) as view_distirbute:
                    interface_distribute.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_clientinfo:
                    interface_clientinfo.render()
                with gr.Group(visible=True, elem_id="shared_group") as view_basicmedicalrecord:
                    interface_basicmedicalrecord.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_hospitalguide:
                    interface_hospitalguide.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_hospitalregister:
                    interface_hospitalregister.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_quality_inspect:
                    quality_inspect_interface.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_quality_modify:
                    quality_modify_interface.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_doctormedicalrecord:
                    interface_doctormedicalrecord.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_diagnosis:
                    interface_diagnosis.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_examass:
                    interface_examass.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_scheme:
                    interface_scheme.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_inpatient:
                    interface_inpatient.render()
                with gr.Group(visible=False, elem_id="shared_group") as view_returnvisit:
                    interface_returnvisit.render()

        def navigation(idx, state):
            idx = int(idx)
            state[idx] = not state[idx]
            if idx == 0:
                return state, gr.update(visible=state[idx]), gr.update(visible=state[idx]), \
                    gr.update(visible=state[idx]), gr.update(visible=state[idx])
            if idx == 1:
                return state, gr.update(visible=state[idx]), gr.update(visible=state[idx]), \
                    gr.update(visible=state[idx]), gr.update(visible=state[idx]), gr.update(visible=state[idx]), \
                    gr.update(visible=state[idx]), gr.update(visible=state[idx])
            if idx == 2:
                return state, gr.update(visible=state[idx])

        pre_btn.click(
            fn=navigation,
            inputs=[gr.Text(value=0, visible=False), state],
            outputs=[state, clientinfo_btn, basicmedicalrecord_btn, hospitalguide_btn, hospitalregister_btn]
        )
        in_btn.click(
            fn=navigation,
            inputs=[gr.Text(value=1, visible=False), state],
            outputs=[state, quality_inspect_btn, quality_modify_btn, doctormedicalrecord_btn, diagnosis_btn, examass_btn, scheme_btn, inpatient_btn]
        )
        post_btn.click(
            fn=navigation,
            inputs=[gr.Text(value=2, visible=False), state],
            outputs=[state, returnvisit_btn]
        )

        def sub_btns_click(btns):
            for idx, btn in enumerate(btns):
                btn.click(
                    lambda idx=idx: [gr.update(visible=(i==idx)) for i in range(len(btns))],
                    outputs=[view_clientinfo, view_basicmedicalrecord, view_hospitalguide, view_hospitalregister,
                             view_quality_inspect, view_quality_modify, view_doctormedicalrecord, view_diagnosis,
                             view_examass, view_scheme, view_inpatient, view_returnvisit]
                )

        sub_btns_click(
            btns=[clientinfo_btn, basicmedicalrecord_btn, hospitalguide_btn, hospitalregister_btn, quality_inspect_btn,
                  quality_modify_btn, doctormedicalrecord_btn, diagnosis_btn, examass_btn, scheme_btn, inpatient_btn, returnvisit_btn]
        )

    return app

if __name__ == "__main__":
    app = build_app()
    app.queue(
        default_concurrency_limit=args.concurrency_count,
        api_open=True,
        status_update_rate=10,
    ).launch(
        server_name=args.host,
        server_port=args.gradio_port,
        share=args.share,
        max_threads=200,
        auth=None,
        ssl_certfile="./cert.pem",
        ssl_keyfile="./key.pem",
        ssl_verify=False,
    )
