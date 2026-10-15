import { Alert, Space, Tag, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import type { evaluationOutputs } from './evaluationOutputs';
import { experimentLabel } from './displayLabels';

export default function ExperimentObservedOutputs({ groups }: { groups: ReturnType<typeof evaluationOutputs> }) {
    const { t, i18n } = useTranslation();
    if (!groups.length) return null;
    return <div className="p-4">
        <Alert type="info" showIcon className="mb-3" message={t('experimentRecords.observed-outputs-description')} />
        {groups.map(group => <section key={group.attemptId} className="mb-3">
            <Space wrap>
                <Typography.Text strong>{t('experimentRecords.execution')} #{group.attemptNumber}</Typography.Text>
                {group.currentStage && <Typography.Text type="secondary">
                    {t('experimentRecords.current-stage')}: {group.currentStage.split('、').map(stage => experimentLabel(t, 'stage-labels', stage)).join(t('experimentRecords.list-separator'))}
                </Typography.Text>}
            </Space>
            {group.items.map(item => <article key={item.path} className="experiment-asset-card">
                <Space wrap>
                    <Typography.Text strong title={item.name}>{experimentLabel(t, 'stage-labels', item.name)}</Typography.Text>
                    <Tag color={item.archivedVersionId ? 'green' : item.exists ? 'blue' : 'default'}>
                        {t(`experimentRecords.observed-output-status.${item.archivedVersionId ? 'archived'
                            : item.exists ? 'observed' : item.status === 'running' ? 'running' : 'pending'}`)}
                    </Tag>
                    {item.logError && <Tag color="orange">{t('experimentRecords.stage-log-error')}</Tag>}
                </Space>
                {item.scores.length > 0 && <div className="mt-2"><Space wrap>
                    {item.scores.map(([name, value]) => <Typography.Text key={name}>
                        {t(`experimentRecords.observed-score.${name}`)}: <strong>
                            {value.toLocaleString(i18n.language, { maximumFractionDigits: 4 })}
                        </strong>
                    </Typography.Text>)}
                </Space></div>}
                <div className="mt-2 text-xs">
                    <Typography.Text type="secondary">{t('experimentRecords.output-file-path')}: </Typography.Text>
                    <Typography.Text className="break-all" copyable>{item.path}</Typography.Text>
                </div>
                {item.logPath && <div className="mt-1 text-xs">
                    <Typography.Text type="secondary">{t('experimentRecords.stage-log-path')}: </Typography.Text>
                    <Typography.Text className="break-all" copyable>{item.logPath}</Typography.Text>
                </div>}
            </article>)}
        </section>)}
    </div>;
}
