import { useState } from 'react';
import { Alert, Descriptions, Input, Select, Space, Table, Tag, Typography } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { filterBenchmarkSamples, type BenchmarkReport, type BenchmarkSample } from './benchmarkReport';

export default function BenchmarkDetails({ report }: { report: BenchmarkReport }) {
    const { t, i18n } = useTranslation();
    const [search, setSearch] = useState('');
    const [status, setStatus] = useState('all');
    const [page, setPage] = useState(1);
    const statusLabel = (value: string) => t(`experimentRecords.sample-status.${value}`);
    const statuses = ['correct', 'incorrect', 'failed', 'invalid', 'unscored'] as const;
    const percent = (value: number) => `${(value * 100).toLocaleString(i18n.language, { maximumFractionDigits: 2 })}%`;
    return <>
        {report.groups.length > 0 && <section className="evaluation-section">
            <div className="evaluation-table-heading"><h4 className="evaluation-section-title">{t('experimentRecords.subset-details')}</h4></div>
            {report.groupsTruncated && <Alert className="mb-3" type="info"
                message={t('experimentRecords.subset-preview-limited')} />}
            <Table size="small" rowKey="name" dataSource={report.groups} pagination={{ pageSize: 10, hideOnSinglePage: true }}
                scroll={{ x: 560 }} columns={[
                    { title: t('experimentRecords.subset'), dataIndex: 'name' },
                    { title: t('experimentRecords.samples'), dataIndex: 'samples', align: 'right' },
                    { title: statusLabel('correct'), dataIndex: 'correct', align: 'right' },
                    { title: statusLabel('failed'), dataIndex: 'failed', align: 'right' },
                    { title: t('experimentRecords.metric-labels.accuracy'), align: 'right',
                        render: (_, group) => group.scored ? percent(group.correct / group.samples) : '—' },
                ]} />
        </section>}
        <section className="evaluation-section" aria-label={t('experimentRecords.sample-details')}>
            <div className="evaluation-table-heading"><h4 className="evaluation-section-title">{t('experimentRecords.sample-details')}</h4></div>
            <Space wrap className="mb-3">
                {statuses.map(value => <Tag key={value} color={value === 'failed' ? 'red' : value === 'correct' ? 'green' : undefined}>
                    {statusLabel(value)}: {report.counts[value].toLocaleString(i18n.language)}
                </Tag>)}
            </Space>
            <Alert className="mb-3" type="info" showIcon message={t('experimentRecords.sample-status-help')} />
            {report.truncated && <Alert className="mb-3" type="info" showIcon
                message={t('experimentRecords.sample-preview-limited', { count: report.samples.length, total: report.total })} />}
            <Space wrap className="mb-3">
                <Input prefix={<SearchOutlined />} allowClear value={search}
                    placeholder={t('experimentRecords.search-samples')} aria-label={t('experimentRecords.search-samples')}
                    onChange={event => { setSearch(event.target.value); setPage(1); }} />
                <Select value={status} style={{ minWidth: 160 }} aria-label={t('experimentRecords.sample-status-filter')}
                    onChange={value => { setStatus(value); setPage(1); }}
                    options={['all', ...statuses].map(value => ({ value, label: statusLabel(value) }))} />
            </Space>
            <Table<BenchmarkSample> size="small" rowKey="index" scroll={{ x: 680 }}
                dataSource={filterBenchmarkSamples(report.samples, search, status)}
                pagination={{ current: page, pageSize: 10, showSizeChanger: false, onChange: setPage,
                    showTotal: total => t('experimentRecords.sample-rows', { count: total }) }} columns={[
                    { title: '#', dataIndex: 'index', width: 60, sorter: (a, b) => a.index - b.index },
                    { title: t('experimentRecords.sample-question'), dataIndex: 'question', ellipsis: true },
                    { title: t('experimentRecords.reference-answer'), dataIndex: 'expected', width: 110, ellipsis: true },
                    { title: t('experimentRecords.model-answer'), width: 140, ellipsis: true,
                        render: (_, sample) => sample.predicted || sample.response || '—' },
                    { title: t('experimentRecords.status-label'), dataIndex: 'status', width: 120,
                        render: value => <Tag color={value === 'failed' ? 'red' : value === 'correct' ? 'green' : undefined}>{statusLabel(value)}</Tag> },
                ]} expandable={{ expandedRowRender: sample => <>
                    <Descriptions size="small" column={1} items={[
                        { key: 'question', label: t('experimentRecords.sample-question'), children: <div className="whitespace-pre-wrap break-all">{sample.question || '—'}</div> },
                        { key: 'expected', label: t('experimentRecords.reference-answer'), children: sample.expected || '—' },
                        { key: 'response', label: t('experimentRecords.model-response'), children: <div className="max-h-80 overflow-auto whitespace-pre-wrap break-all">{sample.response || sample.predicted || '—'}</div> },
                        ...Object.entries(sample.metrics).map(([name, value]) => ({ key: name,
                            label: t(`experimentRecords.metric-labels.${name}`, { defaultValue: name }), children: value.toLocaleString(i18n.language) })),
                    ]} />
                    {sample.error && <Alert type="error" message={sample.error} />}
                </> }} />
            <Typography.Paragraph type="secondary" className="mt-2">{t('experimentRecords.full-report-in-artifacts')}</Typography.Paragraph>
        </section>
    </>;
}
