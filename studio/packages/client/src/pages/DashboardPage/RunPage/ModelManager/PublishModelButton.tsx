import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Form, Input, Modal, Progress, Select, Spin, Tag } from 'antd';
import { Upload } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { trpc } from '@/api/trpc';
import type { ModelInfo, ModelPublication } from '@shared/types';
import { ExperimentLink } from '../ExperimentManager/SourceButton';

type Destination = { destinationContainer: string; destinationDirectory: string; destinationName: string };
const isActive = (job: ModelPublication) => !['finished', 'failed'].includes(job.status);

function createRequestKey() {
    // getRandomValues also works when Studio is opened over HTTP on a LAN.
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    return `pub-${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export default function PublishModelButton({ model, container }: { model: ModelInfo; container: string }) {
    const { t } = useTranslation();
    const [open, setOpen] = useState(false);
    const [accepted, setAccepted] = useState<ModelPublication | null>(null);
    const [form] = Form.useForm<Destination>();
    const request = useRef<{ signature: string; key: string } | null>(null);
    const sourceContainer = model.containerName || container;
    const source = { nodeId: model.nodeId || '', container: sourceContainer, modelType: model.type || '',
        modelName: model.name, modelPath: model.path || '' };
    const available = Boolean(source.nodeId && source.modelType && source.modelPath);
    const targets = trpc.getModelPublicationTargets.useQuery({ nodeId: source.nodeId },
        { enabled: open && available, staleTime: 0, retry: 1 });
    const publications = trpc.getModelPublications.useQuery(source,
        { enabled: open && available, staleTime: 0, refetchInterval: open ? 3000 : false, retry: 1 });
    const publish = trpc.publishModelToInference.useMutation();
    const jobs = [...(publications.data || [])];
    if (accepted && !jobs.some(job => job.requestKey === accepted.requestKey)) jobs.unshift(accepted);
    const running = jobs.some(isActive);

    useEffect(() => {
        if (open && targets.data) {
            if (!form.isFieldTouched('destinationContainer')) {
                const selected = targets.data.containers.includes(targets.data.defaultContainer)
                    ? targets.data.defaultContainer : targets.data.containers.find(name => name !== sourceContainer);
                form.setFieldValue('destinationContainer', selected);
            }
            if (!form.isFieldTouched('destinationDirectory')) form.setFieldValue('destinationDirectory', targets.data.directory);
            if (!form.isFieldTouched('destinationName')) form.setFieldValue('destinationName', model.name);
        }
    }, [open, targets.data, form, model.name, sourceContainer]);

    const submit = async (values: Destination) => {
        const signature = JSON.stringify({ ...source, ...values });
        const previous = jobs.find(job => job.requestKey === request.current?.key);
        if (!request.current || request.current.signature !== signature || (previous && !isActive(previous))) {
            request.current = { signature, key: createRequestKey() };
        }
        try {
            const job = await publish.mutateAsync({ ...source, ...values, requestKey: request.current.key });
            setAccepted(job);
            await publications.refetch();
        } catch {
            // Keep the request key after a lost response, so a retry cannot launch another copy.
            void publications.refetch();
        }
    };

    const retry = (job: ModelPublication) => {
        const separator = job.destination.lastIndexOf('/');
        form.setFieldsValue({ destinationContainer: job.destinationContainer,
            destinationDirectory: job.destination.slice(0, separator), destinationName: job.destination.slice(separator + 1) });
        request.current = null;
        publish.reset();
    };

    return <>
        <Button type="link" size="small" icon={<Upload size={13} />} disabled={!available}
            title={!available ? t('modelPublication.refresh-source') : undefined}
            onClick={() => setOpen(true)}>{t('modelPublication.publish')}</Button>
        <Modal open={open} title={t('modelPublication.title')} footer={null} width={620} onCancel={() => setOpen(false)}>
            <p className="mb-3 break-all text-xs text-muted-foreground">{model.nodeName || model.nodeId} · {sourceContainer}:{model.path}/{model.name}</p>
            <Alert type="info" showIcon className="mb-4" message={t('modelPublication.description')} />
            {model.artifactTypes?.includes('adapter') && <Alert type="warning" showIcon className="mb-4"
                message={t('modelPublication.adapter')} />}
            {targets.data && targets.data.containers.length === 0 && <Alert type="warning" showIcon className="mb-4"
                message={t('modelPublication.no-containers')} />}
            {targets.isLoading ? <Spin /> : targets.error ? <Alert type="error" message={targets.error.message}
                action={<Button onClick={() => targets.refetch()}>{t('modelPublication.refresh')}</Button>} /> :
                <Form form={form} layout="vertical" onFinish={submit} disabled={publish.isPending || running}>
                    <Form.Item name="destinationContainer" label={t('modelPublication.container')}
                        rules={[{ required: true, message: t('modelPublication.choose-container') }]}>
                        <Select options={targets.data?.containers.map(name => ({ label: name, value: name }))}
                            placeholder={t('modelPublication.choose-container')} />
                    </Form.Item>
                    <Form.Item name="destinationDirectory" label={t('modelPublication.directory')}
                        extra={t('modelPublication.directory-help')}
                        rules={[{ required: true }, { pattern: /^\/(?!.*(?:^|\/)\.\.(?:\/|$))(?!.*\\).+$/, message: t('modelPublication.invalid-directory') }]}>
                        <Input disabled={!targets.data?.canChooseDirectory || publish.isPending || running} />
                    </Form.Item>
                    <Form.Item name="destinationName" label={t('modelPublication.name')}
                        rules={[{ required: true }, { pattern: /^[A-Za-z0-9][A-Za-z0-9_.-]*$/, message: t('modelPublication.invalid-name') }, { max: 200 }]}>
                        <Input />
                    </Form.Item>
                    <Button type="primary" htmlType="submit" loading={publish.isPending}
                        disabled={running || !targets.data?.containers.length || publications.isLoading || Boolean(publications.error)}>
                        {t('modelPublication.publish')}
                    </Button>
                </Form>}
            {publish.error && <Alert className="mt-3" type="error" showIcon message={publish.error.message} />}
            {publications.error && <Alert className="mt-3" type="error" message={publications.error.message} />}
            <div className="mt-4 flex items-center justify-between">
                <span className="font-medium">{t('modelPublication.history')}</span>
                <Button size="small" loading={publications.isFetching} onClick={() => publications.refetch()}>{t('modelPublication.refresh')}</Button>
            </div>
            {jobs.map(job => <div key={job.requestKey} className="mt-3 rounded-lg border border-border p-3">
                <div className="flex items-center justify-between gap-2">
                    <span className="break-all text-xs">{job.destinationContainer}:{job.destination}</span>
                    <Tag color={job.status === 'finished' ? 'green' : job.status === 'failed' ? 'red' : 'blue'}>
                        {t(`modelPublication.status.${job.status}`)}
                    </Tag>
                </div>
                <Progress percent={job.progress_percent} size="small" status={job.status === 'failed' ? 'exception' : job.status === 'finished' ? 'success' : 'active'} />
                <p className="text-xs text-muted-foreground">{job.error || job.message}</p>
                {job.status === 'finished' && <p className="mt-2 text-xs">{t('modelPublication.ready', {
                    directory: job.destination.slice(0, job.destination.lastIndexOf('/')), name: job.destination.split('/').pop(),
                })}</p>}
                <div className="mt-2 flex gap-2">
                    <ExperimentLink experimentId={job.experiment_id || undefined} />
                    {job.status === 'failed' && <Button size="small" disabled={running || publish.isPending} onClick={() => retry(job)}>
                        {t('modelPublication.retry')}
                    </Button>}
                </div>
            </div>)}
        </Modal>
    </>;
}
