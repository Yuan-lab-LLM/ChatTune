import { useState } from 'react';
import { Button, DatePicker, Input, Select, Space } from 'antd';
import type { Dayjs } from 'dayjs';
import { useTranslation } from 'react-i18next';
import { experimentStatusSchema, taskKindSchema } from '@shared/experiment';
import type { ExperimentListFilters } from '@shared/experiment';

export default function ExperimentFilters({ onApply }: { onApply: (filters: ExperimentListFilters) => void }) {
    const { t } = useTranslation();
    const [search, setSearch] = useState('');
    const [project, setProject] = useState('');
    const [tag, setTag] = useState('');
    const [status, setStatus] = useState<ExperimentListFilters['status']>();
    const [taskKind, setTaskKind] = useState<ExperimentListFilters['taskKind']>();
    const [dates, setDates] = useState<[Dayjs, Dayjs] | null>(null);
    const apply = (searchValue = search) => onApply({
        search: searchValue.trim() || undefined, project: project.trim() || undefined, tag: tag.trim() || undefined, status, taskKind,
        createdFrom: dates?.[0].startOf('day').toISOString(),
        createdBefore: dates?.[1].add(1, 'day').startOf('day').toISOString(),
    });
    const reset = () => {
        setSearch(''); setProject(''); setTag(''); setStatus(undefined); setTaskKind(undefined); setDates(null);
        onApply({});
    };
    return <div className="experiment-filters">
        <Input.Search size="small" className="experiment-filter-search" value={search} maxLength={200} allowClear
            aria-label={t('experimentRecords.search-all-by-name-or-id')}
            placeholder={t('experimentRecords.search-all-by-name-or-id')}
            onChange={event => setSearch(event.target.value)} onSearch={value => apply(value)} />
        <Select size="small" value={status} allowClear
            aria-label={t('experimentRecords.filter-status')} placeholder={t('experimentRecords.all-statuses')}
            options={experimentStatusSchema.options.map(value => ({ value, label: t(`experimentRecords.status.${value}`) }))}
            onChange={setStatus} />
        <Select size="small" value={taskKind} allowClear
            aria-label={t('experimentRecords.filter-task-kind')} placeholder={t('experimentRecords.all-task-kinds')}
            options={taskKindSchema.options.map(value => ({ value, label: t(`experimentRecords.kind.${value}`) }))}
            onChange={setTaskKind} />
        <Input size="small" className="experiment-filter-project" value={project} maxLength={200} allowClear
            aria-label={t('experimentRecords.filter-project')} placeholder={t('experimentRecords.filter-project')}
            onChange={event => setProject(event.target.value)} onPressEnter={() => apply()} />
        <Input size="small" className="experiment-filter-project" value={tag} maxLength={40} allowClear aria-label={t('experimentRecords.filter-tag')}
            placeholder={t('experimentRecords.filter-tag')} onChange={event => setTag(event.target.value)} onPressEnter={() => apply()} />
        <DatePicker.RangePicker size="small" className="experiment-filter-dates" value={dates} aria-label={t('experimentRecords.filter-created-range')}
            placeholder={[t('experimentRecords.created-from'), t('experimentRecords.created-to')]}
            onChange={range => setDates(range?.[0] && range[1] ? [range[0], range[1]] : null)} />
        <Space size={6} className="experiment-filter-actions">
            <Button size="small" type="primary" onClick={() => apply()}>{t('experimentRecords.apply-filters')}</Button>
            <Button size="small" onClick={reset}>{t('experimentRecords.reset-filters')}</Button>
        </Space>
    </div>;
}
