import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, Card, Alert } from 'antd';
import { Play, Terminal, CheckCircle, Send } from 'lucide-react';

interface Step3TrainingProps {
  isExecuting: boolean;
  logs: string[];
  onFillCommand: () => Promise<boolean>;
  isFilled: boolean;
}

export function Step3Training({ isExecuting, logs, onFillCommand, isFilled }: Step3TrainingProps) {
  const { t } = useTranslation();
  const [isChecking, setIsChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 命令始终使用中文，不翻译
  const command = '运行lora批量训练';

  const handleConfirm = async () => {
    setIsChecking(true);
    setError(null);

    try {
      const success = await onFillCommand();

      if (!success) {
        setError(t('wizard.validation.noRunInstance'));
      }
    } catch (err) {
      console.error('[Step3Training] Error during confirm:', err);
      setError(t('wizard.validation.stepFailed'));
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div className="p-3 bg-green-100 rounded-2xl shrink-0 border border-green-200/60">
          <Play className="w-8 h-8 text-green-600" />
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-semibold mb-2">{t('wizard.step3.title')}</h4>
          <p className="text-muted-foreground">{t('wizard.step3.subtitle')}</p>
        </div>
      </div>

      <Card
        title={
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-green-600" />
            <span className="font-semibold">{t('wizard.step3.commandPreview')}</span>
          </div>
        }
        className="rounded-2xl border-border/25 shadow-none bg-muted/10"
        size="small"
      >
        <div className="mt-2">
          <code className="block bg-slate-950 text-emerald-300 p-4 rounded-2xl text-sm font-mono border border-slate-800">{command}</code>
        </div>
        <div className="mt-3 grid gap-2 rounded-2xl border border-emerald-100 bg-emerald-50/70 p-3 text-xs text-slate-600">
          <div className="font-medium text-slate-800">{t('wizard.commandTeaching.title')}</div>
          <div>{t('wizard.commandTeaching.training.task')}</div>
          <div>{t('wizard.commandTeaching.training.mode')}</div>
        </div>
      </Card>

      {!isFilled ? (
        <>
          {error && (
            <Alert
              message={t('wizard.step3.error')}
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
              className="rounded-2xl border-red-200/70 bg-red-50/70"
            />
          )}

          <Alert
            message={t('wizard.step3.instruction.title')}
            description={
              <div className="space-y-2">
                <p>{t('wizard.step3.instruction.step1')}</p>
                <p>{t('wizard.step3.instruction.step2')}</p>
                <p>{t('wizard.step3.instruction.step3')}</p>
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
              icon={<Send className="w-5 h-5" />}
              onClick={handleConfirm}
              loading={isChecking}
              className="h-10 rounded-xl px-5 bg-green-600 hover:bg-green-700"
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

      {logs.length > 0 && (
        <Card title={t('wizard.step4.logs')} size="small" className="rounded-2xl border-border/25 shadow-none bg-muted/10">
          <div className="max-h-32 overflow-y-auto font-mono text-xs space-y-1 rounded-xl bg-background border border-border/20 p-3">
            {logs.map((log, idx) => (
              <div key={idx} className="text-muted-foreground">{log}</div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
