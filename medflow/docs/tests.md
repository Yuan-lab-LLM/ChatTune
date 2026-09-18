## 脚本说明

### 兼容冒烟测试

医疗兼容冒烟测试覆盖预问诊、诊断、住院文书、随访和病历质控。测试脚本使用自身所在目录解析数据文件，可以从任意工作目录运行；响应临时文件会在每个脚本结束时自动删除。

```bash
bash tests/run-medical-smoke.sh <医疗服务地址> <医疗服务端口>
```

也可以通过 `MEDFLOW_BASE_URL` 指定完整地址。Case2Chat 是独立服务，不会默认运行；需要验证时设置 `MEDFLOW_SMOKE_INCLUDE_CASE2CHAT=1`，并通过第三、第四个参数或 `CASE2CHAT_BASE_URL` 提供地址：

```bash
MEDFLOW_SMOKE_INCLUDE_CASE2CHAT=1 \
  bash tests/run-medical-smoke.sh <医疗服务地址> <医疗服务端口> <Case2Chat地址> <Case2Chat端口>
```

这些脚本只校验 HTTP 状态和稳定的 JSON 结构，不比较模型生成的具体医疗文本。其他脚本继续作为手工业务样例使用。

### 手工业务脚本

| 功能       | 测试脚本                                                     | 说明                                                         | 是否对话 |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 患者建档   | [test-clientinfo.sh](../tests/test-clientinfo.sh)            | 收集患者的姓名、证件号码、联系方式等，生成建档信息，带有基础的校验功能。 | Y        |
| 导诊       | [test-hospitalguide.sh](../tests/test-hospitalguide.sh)       | 简单了解患者的病情，并为患者指引科室。                       | Y        |
| 预问诊报告 | [test-basicmedicalrecord.sh](../tests/test-basicmedicalrecord.sh) | 收集患者的主诉、现病史、既往史、个人史、过敏史等，生成预问诊报告。 | Y        |
| 挂号       | [test-register.sh](../tests/test-register.sh)                | 理解患者的意图，为患者推荐合适的挂号信息。                   | Y        |
| 诊断       | [test-diagnosis.sh](../tests/test-diagnosis.sh)              | 根据预问诊报告，生成初步诊断结果。                           | N        |
| 检查和化验 | [test-examineassay.sh](../tests/test-examineassay.sh)        | 根据预问诊报告和诊断结果，生成检查单和化验单。               | N        |
| 治疗方案   | [test-therapyscheme.sh](../tests/test-therapyscheme.sh)      | 根据预问诊报告和诊断结果，生成治疗方案。                     | N        |
| 复诊       | [test-returnvisit.sh](../tests/test-returnvisit.sh)          | 根据上次就诊记录，向患者了解治疗效果，判断是否需要复诊。     | Y        |
| 病历       | [test-doctormedicalrecord.sh](../tests/test-doctormedicalrecord.sh) | 根据医生的描述，AI对病历进行修改，支持普通病历与模板病历的修改。 | N        |

