import { useEffect, useMemo, useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { Modal, Steps, Button, message } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Database, Settings, Play, LineChart, ChevronRight, Zap, CheckCircle } from 'lucide-react';
import { useWizardState } from './hooks/useWizardState';
import { Step1DatasetIntro } from './steps/Step1DatasetIntro';
import { Step2Preprocess } from './steps/Step2Preprocess';
import { Step3Training } from './steps/Step3Training';
import { Step4Monitor } from './steps/Step4Monitor';
import type { QuickStartWizardProps, QuickStartWizardRef } from './types';

export const QuickStartWizard = forwardRef<QuickStartWizardRef, QuickStartWizardProps>(function QuickStartWizard({
  open,
  onClose,
  onTabChange,
  onSendCommand,
  queryState,
  datasets,
  selectedDataset,
  onSelectDataset,
  requiredDatasetId = 'medical-example',
  inputRequests,
  runs,
  focusOnLatestRun,
  setFocusOnLatestRun,
  onAsChatSendComplete,
  onStepChange
}: QuickStartWizardProps, ref) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { state, selectDataset: selectWizardDataset, nextStep, addLog, setExecuting, reset } = useWizardState();

  // 步骤配置 - 使用翻译
  const STEP_ITEMS = useMemo(() => [
    { title: t('wizard.steps.step1.title'), icon: <Database className="w-5 h-5" /> },
    { title: t('wizard.steps.step2.title'), icon: <Settings className="w-5 h-5" /> },
    { title: t('wizard.steps.step3.title'), icon: <Play className="w-5 h-5" /> },
    { title: t('wizard.steps.step4.title'), icon: <LineChart className="w-5 h-5" /> },
  ], [t]);

  useEffect(() => {
    onStepChange?.(
      state.currentStep,
      String(STEP_ITEMS[state.currentStep]?.title || ''),
    );
  }, [STEP_ITEMS, onStepChange, state.currentStep]);

  // 使用 ref 存储 inputRequests 的最新值，避免闭包问题
  const inputRequestsRef = useRef(inputRequests);
  useEffect(() => {
    inputRequestsRef.current = inputRequests;
  }, [inputRequests]);

  // 命令填充状态 - 从 sessionStorage 恢复
  const [isCommandFilled, setIsCommandFilled] = useState(() => {
    if (typeof window === 'undefined') return false;
    return sessionStorage.getItem('wizard_command_filled') === 'true';
  });
  const [filledCommand, setFilledCommand] = useState(() => {
    if (typeof window === 'undefined') return '';
    return sessionStorage.getItem('wizard_filled_command') || '';
  });
  // 第三步命令填充状态
  const [isTrainingCommandFilled, setIsTrainingCommandFilled] = useState(() => {
    if (typeof window === 'undefined') return false;
    return sessionStorage.getItem('wizard_training_command_filled') === 'true';
  });
  // 期望命令状态（用于判定步骤完成）
  const [expectedCommand, setExpectedCommand] = useState<string | null>(null);
  // 向导完成状态
  const [isWizardCompleted, setIsWizardCompleted] = useState(() => {
    if (typeof window === 'undefined') return false;
    return sessionStorage.getItem('wizard_completed') === 'true';
  });

  // 重新开始向导
  const handleRestartWizard = () => {
    // 清除所有状态
    sessionStorage.removeItem('wizard_completed');
    sessionStorage.removeItem('wizard_step');
    sessionStorage.removeItem('wizard_command_filled');
    sessionStorage.removeItem('wizard_filled_command');
    sessionStorage.removeItem('wizard_training_command_filled');
    sessionStorage.removeItem('wizard_query_state');
    sessionStorage.setItem('wizard_active', 'true');
    sessionStorage.setItem('wizard_step', '0');

    // 重置状态
    reset();
    setIsWizardCompleted(false);
    setIsCommandFilled(false);
    setFilledCommand('');
    setIsTrainingCommandFilled(false);
    setExpectedCommand(null);

  };

  // 暴露方法给父组件
  useImperativeHandle(ref, () => ({
    restart: handleRestartWizard,
    handleStepComplete: (sentCommand: string) => {

      // 定义每个步骤期望的命令关键词
      const expectedCommands: Record<number, string[]> = {
        1: ['执行数据预处理', '预处理', '数据预处理'],
        2: ['运行lora', 'lora', '训练', '批量训练']
      };

      const currentStep = state.currentStep;
      const expected = expectedCommands[currentStep];

      if (!expected) {
        return;
      }

      // 验证命令是否包含期望的关键词
      const isValid = expected.some(keyword =>
        sentCommand.toLowerCase().includes(keyword.toLowerCase())
      );

      if (!isValid) {
        message.warning(t('wizard.validation.commandMismatch', {
          step: currentStep + 1,
          keywords: expected.join(t('common.comma') || '、')
        }));
        return;
      }


      if (currentStep === 3) {
        handleCompleteWizard();
        return;
      }

      // 其他步骤正常前进
      nextStep();
    }
  }), [handleRestartWizard, handleCompleteWizard, state.currentStep, nextStep]);

  useEffect(() => {
    if (open) {
      // 注意：向导完成状态的检查现在放在渲染逻辑中
      // 这里只处理未完成时的逻辑

      // 保存向导状态到 sessionStorage
      sessionStorage.setItem('wizard_active', 'true');
      sessionStorage.setItem('wizard_step', String(state.currentStep));
      sessionStorage.setItem('wizard_query_state', queryState);

      // 从 sessionStorage 恢复步骤（解决组件未重新挂载导致的步骤丢失问题）
      const savedStep = sessionStorage.getItem('wizard_step');
      if (savedStep) {
        const stepNum = parseInt(savedStep, 10);
        if (stepNum !== state.currentStep) {
          // 使用 nextStep 多次前进到保存的步骤
          const diff = stepNum - state.currentStep;
          if (diff > 0) {
            for (let i = 0; i < diff; i++) {
              nextStep();
            }
          }
        }
      }

      if (state.currentStep === 0) {
        onTabChange('datasets');

        // 检查是否有缓存的数据集
        const cachedDatasets = localStorage.getItem('cached_datasets');
        if (cachedDatasets && datasets.length === 0) {
          try {
            const parsed = JSON.parse(cachedDatasets);
            const hasExample = parsed.some((d: DatasetInfo) => d.name === requiredDatasetId);
            if (hasExample) {
              // 有缓存且包含示例数据，自动使用缓存
              onSelectDataset(parsed.find((d: DatasetInfo) => d.name === requiredDatasetId) || null);
              // 这里需要通知外部更新 datasets 状态
            }
          } catch (e) {
            console.error('Failed to parse cached datasets:', e);
          }
        }
      }
    } else {
      // 向导关闭时，不清除状态（可能在查询中）
      // 由外部在合适时机清除
    }
  }, [open, state.currentStep, queryState, onTabChange, datasets, requiredDatasetId, onSelectDataset, nextStep]);

  const handleClose = () => {
    // 如果向导已完成，清除 wizard_active，解绑完成状态和引导状态
    // 这样下次点击"新手引导"会进入学习页面，用户可以选择重新学习或重新开始向导
    if (isWizardCompleted) {
      sessionStorage.removeItem('wizard_active');
    }
    onClose();
    // 注意：不调用 reset，保留状态以便下次恢复（除非已完成）
  };

  // 完成整个向导流程
  function handleCompleteWizard() {
    // 标记向导为已完成
    sessionStorage.setItem('wizard_completed', 'true');
    // 清除 wizard_active，解绑完成状态和引导状态
    // 这样下次点击"新手引导"会进入学习页面
    sessionStorage.removeItem('wizard_active');
    setIsWizardCompleted(true);
    // 不关闭 Modal，让完成界面自动渲染
    // 用户会在完成界面上看到成功提示，然后手动点击"关闭"或"重新开始"
  }

  // 辅助函数：确保有运行实例可用（自动查找并切换）
  const ensureRunInstanceAvailable = async (): Promise<boolean> => {

    // 如果已经有输入请求，直接返回成功
    if (inputRequestsRef.current && inputRequestsRef.current.length > 0) {
      return true;
    }


    // 先切换到 runs tab
    onTabChange('runs');

    // 等待 DOM 更新
    await new Promise(resolve => setTimeout(resolve, 500));

    const rows = Array.from(
      document.querySelectorAll('[data-run-id][data-run-status]'),
    ).filter((row) => {
      const status = (row.getAttribute('data-run-status') || '').toUpperCase();
      return status === 'RUNNING' || status === 'PENDING';
    });

    const runId = rows[0]?.getAttribute('data-run-id') || null;

    if (!runId) {
      console.error('No running run found in DOM');
      message.error(t('wizard.validation.noRunInstance'));
      return false;
    }

    // 使用 focusOnLatestRun 机制
    setFocusOnLatestRun(true);

    // 显示提示
    message.info(t('wizard.validation.foundInstance'));

    // 等待 inputRequests 更新（最多等待 5 秒）
    let attempts = 0;
    const maxAttempts = 50; // 50 * 100ms = 5秒

    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;

      // 检查是否已经有 inputRequests（使用 ref 获取最新值）
      if (inputRequestsRef.current && inputRequestsRef.current.length > 0) {
        break;
      }
    }

    // 重置 focusOnLatestRun
    setFocusOnLatestRun(false);

    // 如果等待后仍然没有 inputRequests，提示用户再次点击
    if (!inputRequestsRef.current || inputRequestsRef.current.length === 0) {
      message.warning(t('wizard.validation.connecting'));
      return false;
    }

    return true;
  };

  // 处理确认并填入命令（第二步）
  const handleConfirmAndFill = async (): Promise<boolean> => {

    // 确保有运行实例可用
    const hasRunInstance = await ensureRunInstanceAvailable();
    if (!hasRunInstance) {
      return false;
    }

    // 命令始终使用中文，不翻译
    const command = `执行数据预处理，数据类型为sft，数据格式为diagnosis，数据在/home/workspace/dataset/${selectedDataset?.name}`;

    // 保存命令并标记为已填入
    setFilledCommand(command);
    setIsCommandFilled(true);
    // 保存到 sessionStorage（确保是中文）
    sessionStorage.setItem('wizard_command_filled', 'true');
    sessionStorage.setItem('wizard_filled_command', command);

    // 调用外部回调，将命令填入 AsChat 输入框
    onSendCommand(command);

    // 添加日志
    addLog(`✅ ${t('wizard.status.log.commandFilled')}`);

    return true;
  };

  // 处理第三步：填入训练命令
  const handleFillTrainingCommand = async (): Promise<boolean> => {

    // 确保有运行实例可用
    const hasRunInstance = await ensureRunInstanceAvailable();
    if (!hasRunInstance) {
      return false;
    }

    // 命令始终使用中文，不翻译
    const command = '运行lora批量训练';

    // 保存命令并标记为已填入
    setIsTrainingCommandFilled(true);
    // 保存到 sessionStorage（确保是中文）
    sessionStorage.setItem('wizard_training_command_filled', 'true');
    sessionStorage.setItem('wizard_filled_command', command);

    // 调用外部回调，将命令填入 AsChat 输入框
    onSendCommand(command);

    // 添加日志
    addLog(`✅ ${t('wizard.status.log.trainingCommandFilled')}`);

    return true;
  };

  const handleSendCommand = async (command: string, stepName: string) => {
    setExecuting(true);
    addLog(t('wizard.status.log.stepStarted', { stepName }));

    try {
      onSendCommand(command);
      addLog(t('wizard.status.log.commandSent', { command }));
      await new Promise(resolve => setTimeout(resolve, 1500));
      addLog(t('wizard.status.log.stepCompleted', { stepName }));
      return true;
    } catch (error) {
      addLog(t('wizard.status.log.stepFailed', { error: String(error) }));
      return false;
    } finally {
      setExecuting(false);
    }
  };

  // 处理数据集选择
  const handleSelectDataset = (dataset: DatasetInfo) => {
    // 只有示例数据集才能选择
    if (dataset.name === requiredDatasetId) {
      onSelectDataset(dataset);
      selectWizardDataset({
        id: dataset.name,
        name: dataset.name,
        type: 'raw',
        description: t('wizard.step1.exampleDesc'),
        path: dataset.name,
        format: dataset.format || 'json',
        recordCount: dataset.recordCount || 0
      });
    }
  };

  // 判断是否可以选择下一步
  const canProceedToNext = useMemo(() => {
    if (state.currentStep === 0) {
      // 第一步：必须选择了示例数据集
      return selectedDataset?.name === requiredDatasetId;
    }
    return state.canProceed;
  }, [state.currentStep, state.canProceed, selectedDataset, requiredDatasetId]);

  const renderStepContent = () => {
    switch (state.currentStep) {
      case 0:
        return (
          <Step1DatasetIntro
            queryState={queryState}
            datasets={datasets}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            requiredDatasetId={requiredDatasetId}
          />
        );
      case 1:
        return selectedDataset ? (
          <Step2Preprocess
            selectedDataset={selectedDataset}
            onConfirmAndFill={handleConfirmAndFill}
            isFilled={isCommandFilled}
          />
        ) : null;
      case 2:
        return (
          <Step3Training
            isExecuting={state.isExecuting}
            logs={state.logs}
            onFillCommand={handleFillTrainingCommand}
            isFilled={isTrainingCommandFilled}
          />
        );
      case 3:
        return (
          <Step4Monitor
            logs={state.logs}
          />
        );
      default:
        return null;
    }
  };

  // 如果向导已完成，显示完成界面
  if (isWizardCompleted) {
    return (
      <Modal
        title={null}
        open={open}
        onCancel={handleClose}
      footer={null}
      className="quick-start-wizard-modal"
      width={600}
      centered
      destroyOnHidden
      >
        <div className="px-3 py-5 pt-7">
          <div className="mx-auto max-w-lg rounded-[24px] border border-border/25 bg-muted/10 px-5 py-6">
            <div className="flex justify-center mb-4">
              <div className="p-3 bg-green-100 rounded-full">
                <CheckCircle className="w-16 h-16 text-green-600" />
              </div>
            </div>
            <h3 className="text-xl font-semibold text-center mb-3">{t('wizard.completion.title')}</h3>
            <p className="text-center text-muted-foreground mb-4">{t('wizard.completion.subtitle')}</p>
            <div className="rounded-2xl border border-border/20 bg-background px-4 py-3 text-center text-sm text-muted-foreground space-y-1.5 mb-6">
              <p>✓ {t('wizard.completion.steps.dataPrep')}</p>
              <p>✓ {t('wizard.completion.steps.preprocess')}</p>
              <p>✓ {t('wizard.completion.steps.training')}</p>
              <p>✓ {t('wizard.completion.steps.monitor')}</p>
            </div>
            <div className="flex justify-center gap-3">
              <Button onClick={handleClose} className="h-10 rounded-xl px-5">
                {t('wizard.buttons.close')}
              </Button>
              <Button type="primary" onClick={handleRestartWizard} icon={<CheckCircle className="w-4 h-4" />} className="h-10 rounded-xl px-5">
                {t('wizard.buttons.restart')}
              </Button>
            </div>
          </div>
        </div>
      </Modal>
    );
  }

  return (
    <Modal
      title={null}
      open={open}
      onCancel={handleClose}
      footer={null}
      className="quick-start-wizard-modal"
      width={860}
      centered
      destroyOnHidden
    >
      <div className="px-2 py-3">
        <div className="rounded-[24px] border border-border/25 bg-background overflow-hidden">
          <div className="border-b border-border/20 bg-muted/10 px-4 py-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
                  {t('quickStart.button')}
                </div>
                <h3 className="mt-1 text-lg font-semibold text-foreground">
                  {STEP_ITEMS[state.currentStep]?.title}
                </h3>
              </div>
              <div className="rounded-full border border-border/20 bg-background px-3 py-1 text-xs text-muted-foreground">
                {state.currentStep + 1} / {STEP_ITEMS.length}
              </div>
            </div>

            <div className="mt-3">
              <Steps current={state.currentStep} size="small" responsive className="quick-start-steps">
                {STEP_ITEMS.map((step, index) => (
                  <Steps.Step
                    key={index}
                    title={<span className="text-sm">{step.title}</span>}
                    icon={step.icon}
                  />
                ))}
              </Steps>
            </div>
          </div>

          <div className="min-h-[400px] bg-muted/10 p-4 sm:p-4">{renderStepContent()}</div>

          <div className="flex justify-end items-center px-4 py-3 border-t border-border/20 bg-background/90">

            {state.currentStep === 3 ? (
              <Button
                type="primary"
                onClick={handleCompleteWizard}
                icon={<CheckCircle className="w-4 h-4" />}
                className="h-10 rounded-xl px-5"
              >
                {t('wizard.buttons.complete')}
              </Button>
            ) : (
              <Button
                type="primary"
                onClick={nextStep}
                disabled={!canProceedToNext || state.isExecuting}
                icon={<ChevronRight className="w-4 h-4" />}
                className="h-10 rounded-xl px-5"
              >
                {t('wizard.buttons.next')}
              </Button>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
});

export { Step1DatasetIntro, Step2Preprocess, Step3Training, Step4Monitor };
export { useWizardState };
export * from './types';
export * from './config';
