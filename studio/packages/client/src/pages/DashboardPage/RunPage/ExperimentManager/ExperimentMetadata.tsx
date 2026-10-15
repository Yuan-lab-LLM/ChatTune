import { useState } from 'react';
import { Alert, Button, Input, Select, Space, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import { trpc } from '@/api/trpc';

export default function ExperimentMetadata({ experimentId, notes, tags, onSaved }: {
    experimentId: string; notes: string; tags: string[]; onSaved: () => Promise<unknown>;
}) {
    const { t } = useTranslation();
    const [draftNotes, setDraftNotes] = useState(notes);
    const [draftTags, setDraftTags] = useState(tags);
    const update = trpc.updateExperimentMetadata.useMutation({ onSuccess: async result => {
        setDraftNotes(result.notes); setDraftTags(result.tags); await onSaved();
    } });
    const invalidTags = draftTags.length > 20 || draftTags.some(tag => !tag.trim() || tag.trim().length > 40);
    const dirty = draftNotes !== notes || JSON.stringify(draftTags) !== JSON.stringify(tags);
    return <div className="mb-4 space-y-2">
        <Typography.Text strong>{t('experimentRecords.notes-and-tags')}</Typography.Text>
        <Input.TextArea value={draftNotes} disabled={update.isPending} onChange={event => setDraftNotes(event.target.value)} maxLength={10000}
            autoSize={{ minRows: 2, maxRows: 8 }} aria-label={t('experimentRecords.notes')} placeholder={t('experimentRecords.notes')} />
        <Select mode="tags" style={{ width: '100%' }} value={draftTags} disabled={update.isPending} onChange={setDraftTags}
            aria-label={t('experimentRecords.tags')} placeholder={t('experimentRecords.tags')} />
        {invalidTags && <Alert type="warning" message={t('experimentRecords.tag-limits')} />}
        {update.error && <Alert type="error" message={update.error.message} />}
        <Space><Button size="small" type="primary" loading={update.isPending} disabled={!dirty || invalidTags}
            onClick={() => update.mutate({ experimentId, notes: draftNotes, tags: draftTags })}>{t('experimentRecords.save-metadata')}</Button>
            <Button size="small" disabled={!dirty || update.isPending} onClick={() => { setDraftNotes(notes); setDraftTags(tags); }}>{t('experimentRecords.reset-filters')}</Button>
        </Space>
    </div>;
}
