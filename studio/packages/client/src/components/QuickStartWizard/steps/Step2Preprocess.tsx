import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, Card, Alert, Tag } from 'antd';
import { Database, FileJson, ArrowRight, Terminal, CheckCircle, Play, AlertCircle } from 'lucide-react';
import { PREPROCESS_CONFIG } from '../config';
import type { DatasetInfo } from '@shared/types/trpc';

interface Step2PreprocessProps {
  selectedDataset: DatasetInfo;
  onConfirmAndFill: () => Promise<boolean>;
  isFilled: boolean;
}

export function Step2Preprocess({
  selectedDataset,
  onConfirmAndFill,
  isFilled
}: Step2PreprocessProps) {
  const { t } = useTranslation();
  const [isChecking, setIsChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const outputName = `${selectedDataset.name}${PREPROCESS_CONFIG.outputSuffix}`;
  // 命令始终使用中文，不翻译
  const command = `执行数据预处理，数据类型为${PREPROCESS_CONFIG.targetType}，数据格式为${PREPROCESS_CONFIG.format}，数据在/home/workspace/dataset/${selectedDataset.name}`;

  const handleConfirm = async () => {
    setIsChecking(true);
    setError(null);

    try {
      // 直接调用 onConfirmAndFill，由 QuickStartWizard 处理运行实例检查
      const success = await onConfirmAndFill();

      if (!success) {
        setError(t('wizard.validation.noRunInstance'));
      }
    } catch (err) {
      console.error('[Step2Preprocess] Error during confirm:', err);
      setError(t('wizard.validation.stepFailed'));
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div className="p-3 bg-indigo-100 rounded-2xl shrink-0 border border-indigo-200/60">
          <Terminal className="w-8 h-8 text-indigo-600" />
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-semibold mb-2">{t('wizard.step2.title')}</h4>
          <p className="text-muted-foreground">{t('wizard.step2.subtitle')}</p>
        </div>
      </div>

      <Card className="rounded-2xl border-border/25 shadow-none bg-muted/10">
        <div className="flex items-center justify-center gap-6 py-4">
          <div className="text-center">
            <div className="p-3 bg-blue-100 rounded-2xl inline-block mb-2 border border-blue-200/60">
              <Database className="w-8 h-8 text-blue-600" />
            </div>
            <div className="font-medium">{selectedDataset.name}</div>
            <Tag color="blue">{(selectedDataset.format || 'raw').toUpperCase()}</Tag>
          </div>

          <ArrowRight className="w-8 h-8 text-gray-400" />

          <div className="text-center">
            <div className="p-3 bg-indigo-100 rounded-2xl inline-block mb-2 border border-indigo-200/60">
              <Terminal className="w-8 h-8 text-indigo-600" />
            </div>
            <div className="font-medium">{t('wizard.step2.flow.converting')}</div>
            <Tag color="indigo">{t('wizard.step2.flow.preprocessing')}</Tag>
          </div>

          <ArrowRight className="w-8 h-8 text-gray-400" />

          <div className="text-center">
            <div className="p-3 bg-green-100 rounded-2xl inline-block mb-2 border border-green-200/60">
              <FileJson className="w-8 h-8 text-green-600" />
            </div>
            <div className="font-medium">{outputName}</div>
            <Tag color="green">{t('wizard.step2.flow.output')}</Tag>
          </div>
        </div>
      </Card>

      <Card
        title={
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-blue-600" />
            <span className="font-semibold">{t('wizard.step2.commandPreview')}</span>
          </div>
        }
        className="rounded-2xl border-border/25 shadow-none bg-muted/10"
        size="small"
      >
        <div className="mt-2">
          <code className="block bg-slate-950 text-emerald-300 p-4 rounded-2xl text-sm font-mono whitespace-pre-wrap border border-slate-800">
            {command}
          </code>
        </div>
        <div className="mt-3 grid gap-2 rounded-2xl border border-sky-100 bg-sky-50/70 p-3 text-xs text-slate-600">
          <div className="font-medium text-slate-800">{t('wizard.commandTeaching.title')}</div>
          <div>{t('wizard.commandTeaching.preprocess.task')}</div>
          <div>{t('wizard.commandTeaching.preprocess.dataset')}</div>
          <div>{t('wizard.commandTeaching.preprocess.format')}</div>
        </div>
      </Card>

      {!isFilled ? (
        <>
          {error && (
            <Alert
              message={t('wizard.step2.error')}
              description={error}
              type="error"
              showIcon
              icon={<AlertCircle className="w-5 h-5" />}
              closable
              onClose={() => setError(null)}
              className="rounded-2xl border-red-200/70 bg-red-50/70"
            />
          )}

          <Alert
            message={t('wizard.step2.instruction.title')}
            description={
              <div className="space-y-2">
                <p>{t('wizard.step2.instruction.step1')}</p>
                <p>{t('wizard.step2.instruction.step2')}</p>
                <p>{t('wizard.step2.instruction.step3')}</p>
              </div>
            }
            type="info"
            showIcon
            className="rounded-2xl border-border/25 bg-white/80"
          />

          <div className="flex justify-center">
            <Button
              type="primary"
              size="large"
              icon={<Play className="w-5 h-5" />}
              onClick={handleConfirm}
              loading={isChecking}
              className="h-10 rounded-xl px-5"
            >
              {t('wizard.buttons.generateAndFill')}
            </Button>
          </div>
        </>
      ) : (
        <Alert
          message={t('wizard.status.commandFilled')}
          description={
            <div className="space-y-2">
              <p className="font-medium text-green-700">✓ {t('wizard.status.commandFilledSuccess')}</p>
              <p>{t('wizard.status.clickSendButton')}</p>
              <p className="text-sm text-gray-500">{t('wizard.status.sendButtonHighlighted')}</p>
            </div>
          }
          type="success"
          showIcon
          icon={<CheckCircle className="w-5 h-5" />}
          className="rounded-2xl border-emerald-200/70 bg-emerald-50/70"
        />
      )}
    </div>
  );
}
