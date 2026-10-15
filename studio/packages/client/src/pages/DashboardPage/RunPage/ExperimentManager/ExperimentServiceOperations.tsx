import { Alert, Collapse, Descriptions, Empty, Space, Tag, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import type { ServiceOperationHistory } from '@shared/experiment';

export default function ExperimentServiceOperations({ operations }: { operations: ServiceOperationHistory[] }) {
    const { t, i18n } = useTranslation();
    const time = (value: string | null) => value ? new Date(value).toLocaleString(
        i18n.language.startsWith('zh') ? 'zh-CN' : 'en-US') : '—';
    if (!operations.length) return <Empty description={t('experimentRecords.service-history-empty')} />;
    return <Space direction="vertical" className="w-full" size="middle">
        <Alert type="info" showIcon message={t('experimentRecords.service-history-description')} />
        {operations.map(operation => <article key={operation.id} className="w-full rounded-lg border p-4">
            <Space wrap>
                <Typography.Text strong>{t(operation.cause === 'restart'
                    ? 'experimentRecords.service-restart-stop' : 'experimentRecords.service-stop')}</Typography.Text>
                <Tag color={operation.status === 'succeeded' ? 'green' : operation.status === 'failed' ? 'red' : 'orange'}>
                    {t(`experimentRecords.service-operation-status.${operation.status}`)}
                </Tag>
            </Space>
            <Descriptions className="mt-3" column={1} size="small" items={[
                { key: 'actor', label: t('experimentRecords.service-operation-actor'), children: operation.actorUserId || t('experimentRecords.service-trigger.system') },
                { key: 'trigger', label: t('experimentRecords.service-operation-trigger'), children: t(`experimentRecords.service-trigger.${operation.trigger}`) },
                { key: 'cause', label: t('experimentRecords.service-operation-cause'), children: t(`experimentRecords.service-cause.${operation.cause || 'unknown'}`) },
                { key: 'requested', label: t('experimentRecords.service-operation-requested'), children: time(operation.requestedAt) },
                { key: 'completed', label: t('experimentRecords.service-operation-completed'), children: time(operation.completedAt) },
                { key: 'instance', label: t('experimentRecords.service-instance'), children: <Typography.Text copyable>{operation.serviceInstanceId}</Typography.Text> },
                { key: 'process', label: t('experimentRecords.service-process-release'), children: t(`experimentRecords.service-release.${operation.releaseReady === null ? 'unknown' : operation.releaseReady ? 'confirmed' : 'unconfirmed'}`) },
                { key: 'reservation', label: t('experimentRecords.service-reservation-current'), children: !operation.reservationId
                    ? t('experimentRecords.service-reservation.none')
                    : <Space wrap><Typography.Text copyable>{operation.reservationId}</Typography.Text>
                        <Tag color={operation.reservationStatus === 'released' ? 'green' : 'default'}>
                            {t(`experimentRecords.service-reservation.${operation.reservationStatus || 'unknown'}`, { defaultValue: operation.reservationStatus })}
                        </Tag>{operation.releasedAt && time(operation.releasedAt)}</Space> },
            ]} />
            {operation.result && <Collapse size="small" items={[{ key: 'result', label: t('experimentRecords.service-operation-result'), children: <>
                {operation.returnCode !== null && <p>{t('experimentRecords.exit-code')}: {operation.returnCode}</p>}
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-all text-xs">{operation.result}</pre>
            </> }]} />}
        </article>)}
    </Space>;
}
