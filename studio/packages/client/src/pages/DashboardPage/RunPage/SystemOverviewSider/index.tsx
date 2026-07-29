import { memo, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SystemOverviewData } from '@shared/types/trpc';
import { Users, Clock, CalendarDays, Monitor, AlertCircle, RefreshCw } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';

interface Props {
    data: SystemOverviewData | null;
    onRefreshGPUInfo?: () => void;
}

const SystemOverviewSider = ({ data, onRefreshGPUInfo }: Props) => {
    const { t } = useTranslation();
    const { isAdmin } = useAuth();
    const [isRefreshingGpu, setIsRefreshingGpu] = useState(false);
    const refreshTimeoutRef = useRef<number | null>(null);
    const refreshStartedAtRef = useRef<number | null>(null);
    const latestGpuUpdate = data?.gpuInfo
        ?.map((gpu) => gpu.collectedAt ? new Date(gpu.collectedAt).getTime() : NaN)
        .filter((time) => Number.isFinite(time))
        .reduce<number | null>((latest, time) => latest === null || time > latest ? time : latest, null);
    const latestGpuUpdateText = latestGpuUpdate
        ? new Date(latestGpuUpdate).toLocaleString()
        : '';
    const gpuInfo = data?.gpuInfo || [];
    const visibleGpuInfo = gpuInfo.filter((gpu) => gpu.index >= 0);
    const gpuStatusMessage = gpuInfo.find((gpu) => gpu.index < 0 && gpu.error)?.error;
    const hasStaleGpu = visibleGpuInfo.some((gpu) => gpu.stale);

    useEffect(() => {
        if (!isRefreshingGpu) return;
        if (latestGpuUpdate && latestGpuUpdate !== refreshStartedAtRef.current) {
            if (refreshTimeoutRef.current !== null) {
                window.clearTimeout(refreshTimeoutRef.current);
                refreshTimeoutRef.current = null;
            }
            refreshStartedAtRef.current = null;
            setIsRefreshingGpu(false);
        }
    }, [isRefreshingGpu, latestGpuUpdate]);

    useEffect(() => () => {
        if (refreshTimeoutRef.current !== null) {
            window.clearTimeout(refreshTimeoutRef.current);
        }
    }, []);

    const handleRefreshGPUInfo = () => {
        if (!onRefreshGPUInfo || isRefreshingGpu) return;
        refreshStartedAtRef.current = latestGpuUpdate ?? null;
        setIsRefreshingGpu(true);
        onRefreshGPUInfo();
        if (refreshTimeoutRef.current !== null) {
            window.clearTimeout(refreshTimeoutRef.current);
        }
        refreshTimeoutRef.current = window.setTimeout(() => {
            refreshTimeoutRef.current = null;
            refreshStartedAtRef.current = null;
            setIsRefreshingGpu(false);
        }, 600);
    };

    if (!data) {
        return (
            <div className="flex flex-col items-center justify-center h-full p-5 text-muted-foreground">
                <div className="flex flex-col items-center rounded-2xl border border-border/25 bg-muted/20 px-8 py-7">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-3"></div>
                    <p className="text-sm">{t('overview.loading')}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-2 px-3 pt-1 pb-3 h-full overflow-y-auto">
            {/* 在线用户卡片 */}
            {isAdmin && <div className="rounded-xl border border-border/25 bg-background p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                    <Users className="w-3.5 h-3.5 text-primary" />
                    <span className="text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">{t('overview.online-users')}</span>
                </div>
                <div className="text-2xl font-bold text-foreground leading-tight">
                    {data.onlineUsers}
                </div>
            </div>}

            {/* 系统运行时长 */}
            {isAdmin && <div className="rounded-xl border border-border/25 bg-background p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                    <Clock className="w-3.5 h-3.5 text-primary" />
                    <span className="text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">{t('overview.server-uptime')}</span>
                </div>
                <div className="text-base font-semibold text-foreground leading-snug">
                    {data.serverUptime}
                </div>
            </div>}

            {/* 消息统计 */}
            {isAdmin && <div className="rounded-xl border border-border/25 bg-background p-3">
                <div className="flex items-center gap-1.5 mb-2">
                    <CalendarDays className="w-3.5 h-3.5 text-primary" />
                    <span className="text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">{t('overview.message-stats')}</span>
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                    <div className="text-center p-2 bg-muted/30 rounded-lg border border-border/20">
                        <div className="text-base font-bold text-primary leading-tight">
                            {data.messageStats.today}
                        </div>
                        <div className="text-[10px] text-muted-foreground">{t('overview.today')}</div>
                    </div>
                    <div className="text-center p-2 bg-muted/30 rounded-lg border border-border/20">
                        <div className="text-base font-bold text-primary leading-tight">
                            {data.messageStats.thisWeek}
                        </div>
                        <div className="text-[10px] text-muted-foreground">{t('overview.this-week')}</div>
                    </div>
                    <div className="text-center p-2 bg-muted/30 rounded-lg border border-border/20">
                        <div className="text-base font-bold text-primary leading-tight">
                            {data.messageStats.thisMonth}
                        </div>
                        <div className="text-[10px] text-muted-foreground">{t('overview.this-month')}</div>
                    </div>
                </div>
            </div>}

            {/* GPU 信息 */}
            <div className="rounded-2xl border border-border/25 bg-background p-4">
                <div className="flex items-start justify-between gap-2 mb-3">
                    <div className="flex items-center gap-2 min-w-0">
                        <Monitor className="w-4 h-4 text-primary shrink-0" />
                        <span className="text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">{t('overview.gpu-info')}</span>
                    </div>
                    <div className="flex items-center justify-end gap-1.5">
                        {latestGpuUpdateText && (
                            <span className="text-[10px] text-muted-foreground text-right leading-tight">
                                {t('overview.gpu-last-updated', { time: latestGpuUpdateText })}
                            </span>
                        )}
                        {onRefreshGPUInfo && (
                            <button
                                type="button"
                                onClick={handleRefreshGPUInfo}
                                disabled={isRefreshingGpu}
                                aria-label={t('overview.gpu-refresh')}
                                title={t('overview.gpu-refresh')}
                                className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border/35 bg-background text-muted-foreground transition-colors hover:border-primary/45 hover:text-primary disabled:cursor-not-allowed disabled:opacity-60"
                            >
                                <RefreshCw className={`h-3.5 w-3.5 ${isRefreshingGpu ? 'animate-spin' : ''}`} />
                            </button>
                        )}
                    </div>
                </div>
                {hasStaleGpu && (
                    <div className="mb-3 rounded-lg bg-amber-50 px-2.5 py-1.5 text-[10px] leading-snug text-amber-700">
                        {latestGpuUpdateText ? t('overview.gpu-stale-with-time', { time: latestGpuUpdateText }) : t('overview.gpu-stale')}
                    </div>
                )}
                {visibleGpuInfo.length === 0 ? (
                    <div className="flex items-start gap-2 rounded-xl border border-dashed border-border/40 bg-muted/20 p-3 text-xs text-muted-foreground">
                        <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                        <span>{gpuStatusMessage || t('overview.gpu-refreshing')}</span>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 gap-3">
                        {visibleGpuInfo.map((gpu, index) => {
                            const isAvailable = gpu.available ?? gpu.memoryUsed < 200;
                            // 转换显存单位为 GB
                            const formatMemory = (mb: number) => mb >= 1000 ? `${(mb / 1024).toFixed(1)}GB` : `${mb}MB`;
                            
                            return (
                                <div key={`${gpu.nodeId || 'node'}-${gpu.index}-${index}`} className="p-2.5 bg-muted/25 rounded-xl border border-border/20">
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <div className={`w-2 h-2 rounded-full ${isAvailable ? 'bg-green-500' : 'bg-red-500'}`} />
                                        <span className="text-xs font-medium text-foreground truncate" title={gpu.name}>
                                            {gpu.nodeName ? `${gpu.nodeName} / ` : ''}GPU{gpu.index}
                                        </span>
                                    </div>
                                    <div className="text-[10px] text-muted-foreground mb-1 truncate" title={gpu.name}>
                                        {gpu.name}
                                    </div>
                                    <div className="text-[10px] space-y-0.5">
                                        {isAdmin && (
                                            <div className="flex justify-between gap-2">
                                                <span className="text-muted-foreground">{t('resourceNode.label')}:</span>
                                                <span className="text-foreground truncate" title={gpu.nodeName || gpu.nodeId || '-'}>
                                                    {gpu.nodeName || gpu.nodeId || '-'}
                                                </span>
                                            </div>
                                        )}
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">{t('overview.gpu-memory')}:</span>
                                            <span className="text-foreground">{formatMemory(gpu.memoryUsed)}/{formatMemory(gpu.memoryTotal)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">{t('overview.gpu-utilization')}:</span>
                                            <span className="text-foreground">{gpu.utilization}%</span>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* 更新时间提示 */}
            {isAdmin && <div className="text-xs text-muted-foreground text-center pt-2">
                {t('overview.update-hint')}
            </div>}
        </div>
    );
};

export default memo(SystemOverviewSider);




