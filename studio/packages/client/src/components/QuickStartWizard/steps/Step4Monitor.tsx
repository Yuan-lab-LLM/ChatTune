import { useTranslation } from 'react-i18next';
import { Card, Tag, Timeline } from 'antd';
import { LineChart, Activity, MessageSquare, Eye, Square } from 'lucide-react';

interface Step4MonitorProps {
  logs: string[];
}

export function Step4Monitor({ logs }: Step4MonitorProps) {
  const { t } = useTranslation();

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div className="p-3 bg-purple-100 rounded-2xl shrink-0 border border-purple-200/60">
          <LineChart className="w-8 h-8 text-purple-600" />
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-semibold mb-2">{t('wizard.step4.title')}</h4>
          <p className="text-muted-foreground">{t('wizard.step4.subtitle')}</p>
        </div>
      </div>

      <Card className="rounded-2xl border-border/25 shadow-none bg-muted/10">
        <h5 className="font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-purple-600" />
          {t('wizard.step4.monitorOperations')}
        </h5>

        <Timeline mode="left">
          <Timeline.Item
            dot={<MessageSquare className="w-4 h-4 text-blue-500" />}
            label={<Tag color="blue">{t('wizard.step4.step1.label')}</Tag>}
          >
            <div className="mb-2">
              <strong>{t('wizard.step4.step1.title')}</strong>
              <p className="text-sm text-gray-600">{t('wizard.step4.step1.desc')}</p>
            </div>
          </Timeline.Item>

          <Timeline.Item
            dot={<Eye className="w-4 h-4 text-green-500" />}
            label={<Tag color="green">{t('wizard.step4.step2.label')}</Tag>}
          >
            <div className="mb-2">
              <strong>{t('wizard.step4.step2.title')}</strong>
              <p className="text-sm text-gray-600">{t('wizard.step4.step2.desc')}</p>
            </div>
          </Timeline.Item>

          <Timeline.Item
            dot={<Square className="w-4 h-4 text-red-500" />}
            label={<Tag color="red">{t('wizard.step4.step3.label')}</Tag>}
          >
            <div className="mb-2">
              <strong>{t('wizard.step4.step3.title')}</strong>
              <p className="text-sm text-gray-600">{t('wizard.step4.step3.desc')}</p>
            </div>
          </Timeline.Item>
        </Timeline>
      </Card>

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
