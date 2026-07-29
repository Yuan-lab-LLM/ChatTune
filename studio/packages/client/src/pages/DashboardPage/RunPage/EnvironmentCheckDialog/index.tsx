import { memo } from 'react';
import { Alert, Button, Empty, Input, Modal, Spin, Tag } from 'antd';
import { useTranslation } from 'react-i18next';
import {
    AlertTriangle,
    CheckCircle2,
    Container,
    Cpu,
    Database,
    FileCheck2,
    HelpCircle,
    ShieldCheck,
    XCircle,
    type LucideIcon,
} from 'lucide-react';
import { EnvironmentCheckResult, EnvironmentCheckStatus } from '@shared/types';

interface Props {
    open: boolean;
    containerName: string;
    result: EnvironmentCheckResult | null;
    isChecking: boolean;
    errorMessage?: string;
    onOpenChange: (open: boolean) => void;
    onRunCheck: () => void;
}

const statusConfig: Record<
    EnvironmentCheckStatus,
    {
        tagColor: string;
        iconClassName: string;
        Icon: LucideIcon;
    }
> = {
    ok: {
        tagColor: 'success',
        iconClassName: 'text-green-600',
        Icon: CheckCircle2,
    },
    warning: {
        tagColor: 'warning',
        iconClassName: 'text-amber-600',
        Icon: AlertTriangle,
    },
    error: {
        tagColor: 'error',
        iconClassName: 'text-red-600',
        Icon: XCircle,
    },
};

const itemIconMap: Record<string, LucideIcon> = {
    container: Container,
    gpu: Cpu,
    datasets: Database,
    models: ShieldCheck,
    medicalTests: FileCheck2,
    evaluationResults: FileCheck2,
};

const formatCheckedAt = (checkedAt: string) => {
    if (!checkedAt) return '';
    return new Date(checkedAt).toLocaleString();
};

const EnvironmentCheckDialog = ({
    open,
    containerName,
    result,
    isChecking,
    errorMessage,
    onOpenChange,
    onRunCheck,
}: Props) => {
    const { t } = useTranslation();
    const overall = result ? statusConfig[result.overallStatus] : null;
    const OverallIcon = overall?.Icon || ShieldCheck;
    const isContainerReady =
        result?.items.some(
            (item) => item.key === 'container' && item.status === 'ok',
        ) ?? false;
    const statusLabel = (status: EnvironmentCheckStatus) =>
        t(`environmentCheck.status.${status}`);

    const getItemTitle = (key: string) =>
        t(`environmentCheck.items.${key}.title`);

    const getItemSummary = (key: string, status: EnvironmentCheckStatus) => {
        if (!result) return '';

        switch (key) {
            case 'container':
                return status === 'ok'
                    ? t('environmentCheck.items.container.summaryOk', {
                          container: result.containerName,
                      })
                    : t('environmentCheck.items.container.summaryError', {
                          container: result.containerName,
                      });
            case 'gpu':
                if (result.counts.gpus === 0) {
                    return t('environmentCheck.items.gpu.summaryNone');
                }
                return t('environmentCheck.items.gpu.summary', {
                    available: result.counts.availableGpus,
                    total: result.counts.gpus,
                });
            case 'datasets':
                return result.counts.datasets > 0
                    ? t('environmentCheck.items.datasets.summaryOk', {
                          count: result.counts.datasets,
                      })
                    : t('environmentCheck.items.datasets.summaryEmpty');
            case 'models':
                return result.counts.models > 0
                    ? t('environmentCheck.items.models.summaryOk', {
                          count: result.counts.models,
                      })
                    : t('environmentCheck.items.models.summaryEmpty');
            case 'medicalTests':
                return result.counts.medicalTests > 0
                    ? t('environmentCheck.items.medicalTests.summaryOk', {
                          count: result.counts.medicalTests,
                      })
                    : t('environmentCheck.items.medicalTests.summaryEmpty');
            case 'evaluationResults':
                return result.counts.evaluationResults > 0
                    ? t('environmentCheck.items.evaluationResults.summaryOk', {
                          count: result.counts.evaluationResults,
                      })
                    : t('environmentCheck.items.evaluationResults.summaryEmpty');
            default:
                return '';
        }
    };

    const getItemSuggestion = (key: string, status: EnvironmentCheckStatus) => {
        const suffix = status === 'ok' ? 'suggestionOk' : 'suggestionAction';
        const keyPath = `environmentCheck.items.${key}.${suffix}`;
        const value = t(keyPath);
        return value === keyPath ? '' : value;
    };

    const getItemDetails = (key: string) => {
        if (!result || key !== 'gpu' || result.gpuInfo.length === 0) {
            return '';
        }

        return result.gpuInfo
            .map((gpu) =>
                t('environmentCheck.items.gpu.detailLine', {
                    index: gpu.index,
                    used: gpu.memoryUsed,
                    total: gpu.memoryTotal,
                    utilization: gpu.utilization,
                }),
            )
            .join('\n');
    };

    const summaryCards = result
        ? [
                  {
                      key: 'datasets',
                      label: t('environmentCheck.summary.datasets'),
                      value: isContainerReady ? result.counts.datasets : '—',
                  },
                  {
                      key: 'models',
                      label: t('environmentCheck.summary.models'),
                      value: isContainerReady ? result.counts.models : '—',
                  },
                  {
                      key: 'medicalTests',
                      label: t('environmentCheck.summary.medicalTests'),
                      value: isContainerReady
                          ? result.counts.medicalTests
                          : '—',
                  },
                  {
                      key: 'evaluationResults',
                      label: t('environmentCheck.summary.evaluationResults'),
                      value: isContainerReady
                          ? result.counts.evaluationResults
                          : '—',
                  },
              {
                  key: 'gpu',
                  label: t('environmentCheck.summary.availableGpus'),
                  value: `${result.counts.availableGpus}/${result.counts.gpus}`,
              },
          ]
        : [];

    return (
        <Modal
            title={
                <div className="flex items-center gap-2">
                    <ShieldCheck className="h-5 w-5 text-primary" />
                    <span>{t('environmentCheck.title')}</span>
                </div>
            }
            open={open}
            onCancel={() => onOpenChange(false)}
            footer={
                <div className="flex items-center justify-end">
                    <Button onClick={() => onOpenChange(false)}>
                        {t('environmentCheck.close')}
                    </Button>
                </div>
            }
            width={680}
            className="environment-check-dialog"
        >
            <div className="space-y-3">
                <div className="rounded-xl border border-border/30 bg-muted/10 p-3">
                    <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                        <Container className="h-4 w-4" />
                        {t('environmentCheck.containerLabel')}
                    </div>
                    <div className="flex flex-col gap-2 sm:flex-row">
                        <Input
                            value={containerName}
                            placeholder={containerName}
                            disabled
                        />
                        <Button
                            type="primary"
                            icon={<ShieldCheck className="h-4 w-4" />}
                            loading={isChecking}
                            onClick={onRunCheck}
                            className="sm:w-28"
                        >
                            {result
                                ? t('environmentCheck.rerun')
                                : t('environmentCheck.start')}
                        </Button>
                    </div>
                    <p className="mt-2 text-xs leading-5 text-muted-foreground">
                        {t('environmentCheck.defaultContainerHint', {
                            defaultValue:
                                '这里固定使用后端环境配置中的默认 Docker 容器。如需切换，请在后端配置中修改默认容器。',
                        })}
                    </p>
                </div>

                {errorMessage && (
                    <Alert
                        type="error"
                        showIcon
                        message={t('environmentCheck.failed')}
                        description={errorMessage}
                    />
                )}

                {isChecking && !result ? (
                    <div className="flex items-center justify-center rounded-xl border border-border/25 bg-muted/10 px-6 py-9">
                        <div className="text-center">
                            <Spin size="large" />
                            <p className="mt-3 text-sm text-muted-foreground">
                                {t('environmentCheck.checking')}
                            </p>
                        </div>
                    </div>
                ) : result ? (
                    <>
                        <div className="rounded-xl border border-border/30 bg-background p-3">
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted/60">
                                        <OverallIcon
                                            className={`h-5 w-5 ${overall?.iconClassName || 'text-primary'}`}
                                        />
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm font-semibold">
                                                {t('environmentCheck.resultTitle')}
                                            </span>
                                            {overall && (
                                                <Tag color={overall.tagColor}>
                                                    {statusLabel(result.overallStatus)}
                                                </Tag>
                                            )}
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {result.containerName} · {formatCheckedAt(result.checkedAt)}
                                        </div>
                                    </div>
                                </div>
                                <div className="grid w-full grid-cols-2 gap-1.5 text-center sm:w-[360px] sm:grid-cols-5">
                                    {summaryCards.map((card) => (
                                        <div
                                            key={card.key}
                                            className="rounded-lg bg-muted/25 px-2 py-1.5"
                                        >
                                            <div className="text-base font-semibold leading-5">
                                                {card.value}
                                            </div>
                                            <div className="text-[10px] leading-4 text-muted-foreground">
                                                {card.label}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            {result.items.map((item) => {
                                const config = statusConfig[item.status];
                                const StatusIcon = config.Icon;
                                const ItemIcon = itemIconMap[item.key] || HelpCircle;
                                const details = getItemDetails(item.key);
                                const suggestion = getItemSuggestion(item.key, item.status);

                                return (
                                    <div
                                        key={item.key}
                                        className="rounded-xl border border-border/25 bg-background p-3"
                                    >
                                        <div className="flex items-start gap-2.5">
                                            <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-muted/45">
                                                <ItemIcon className="h-4 w-4 text-muted-foreground" />
                                            </div>
                                            <div className="min-w-0 flex-1">
                                                <div className="flex flex-wrap items-center gap-2">
                                                    <span className="text-sm font-medium">
                                                        {getItemTitle(item.key)}
                                                    </span>
                                                    <Tag color={config.tagColor}>
                                                        {statusLabel(item.status)}
                                                    </Tag>
                                                </div>
                                                <div className="mt-1 flex items-start gap-1.5 text-[13px]">
                                                    <StatusIcon className={`mt-0.5 h-4 w-4 shrink-0 ${config.iconClassName}`} />
                                                    <span>{getItemSummary(item.key, item.status)}</span>
                                                </div>
                                                {details && (
                                                    <pre className="mt-1.5 max-h-20 overflow-auto whitespace-pre-wrap rounded-lg bg-muted/30 p-2 text-[11px] text-muted-foreground">
                                                        {details}
                                                    </pre>
                                                )}
                                                {suggestion && (
                                                    <div className="mt-1.5 rounded-lg border border-primary/10 bg-primary/5 px-2.5 py-1.5 text-xs text-muted-foreground">
                                                        <span className="font-medium text-foreground">
                                                            {t('environmentCheck.suggestionPrefix')}
                                                        </span>
                                                        {suggestion}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </>
                ) : (
                    <div className="rounded-xl border border-border/25 bg-muted/10 px-6 py-8">
                        <Empty
                            image={Empty.PRESENTED_IMAGE_SIMPLE}
                            description={t('environmentCheck.empty')}
                        />
                    </div>
                )}
            </div>
        </Modal>
    );
};

export default memo(EnvironmentCheckDialog);
