import { memo, useMemo, useRef, useCallback, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Legend,
    CartesianGrid,
} from 'recharts';
import { Button } from '@/components/ui/button.tsx';
import { PanelLeftClose, AlertCircle, Camera, Maximize2, Minimize2, AlertTriangle, Sparkles, Activity, MessageSquareText, Loader2, Clock3 } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { ContentType, ContentBlocks, BlockType, SourceType } from '@shared/types/messageForm';
import { Reply } from '@shared/types';
import { useMessageApi } from '@/context/MessageApiContext.tsx';
import html2canvas from 'html2canvas-pro';
import { Textarea } from '@/components/ui/textarea.tsx';
import { Label } from '@/components/ui/label.tsx';

interface Props {
    onClose: () => void;
    replies: Reply[];
    cacheKey?: string;
    initialCache?: TrainingMetricsCacheSnapshot | null;
    onCacheChange?: (snapshot: TrainingMetricsCacheSnapshot) => void;
    onAskAI?: (blocks: ContentBlocks) => void;
    onMonitorTraining?: () => void;
    monitorStatus?: {
        isQuerying: boolean;
        lastQueryAt?: string;
        lastResultAt?: string;
        lastDataAt?: string;
        hasMetrics?: boolean;
        hasNewData?: boolean;
        message?: string;
    };
    isInputDisabled?: boolean;
}

export interface MetricPoint {
    step: number;
    loss: number;
    lr: number;
    elapsedTime?: number;
    remainingTime?: number;
    totalSteps?: number;
}

export interface ProcessData {
    pid: string;
    metrics: MetricPoint[];
    smoothedMetrics: { step: number; loss: number; lr: number }[];
    color: string;
    latestStep: number;
    latestLoss: number;
    latestLR: number;
    latestElapsedTime?: number;
    latestRemainingTime?: number;
    totalSteps?: number;
    progress?: number;
}

export interface TrainingMetricsCacheSnapshot {
    processDataMap: Map<string, ProcessData>;
    selectedPid: string | null;
}

const extractTextFromContent = (content: ContentType): string => {

    if (typeof content === 'string') {
        return content;
    }
    if (Array.isArray(content)) {
        const result = content
            .map((block, index) => {
                if (block.type === 'text' && 'text' in block) {
                    return (block as { text: string }).text || '';
                }
                // 处理 tool_result 类型
                if (block.type === 'tool_result' && 'output' in block) {
                    const output = (block as { output: unknown }).output;
                    if (Array.isArray(output)) {
                        return output.map((o: { text?: string }) => o.text || '').join('');
                    }
                }
                return '';
            })
            .join('');
        return result;
    }
    return '';
};

const generateColor = (pid: string): string => {
    // iOS 风格柔和调色板 - 使用 oklch 颜色空间
    const colors = [
        'oklch(0.55 0.15 250)',  // iOS Blue
        'oklch(0.65 0.12 160)',  // iOS Green
        'oklch(0.65 0.15 30)',   // iOS Orange
        'oklch(0.65 0.2 25)',    // iOS Red
        'oklch(0.6 0.1 200)',    // iOS Teal
        'oklch(0.7 0.12 80)',    // iOS Yellow
        'oklch(0.65 0.15 300)',  // iOS Purple
        'oklch(0.65 0.12 320)',  // iOS Pink
        'oklch(0.55 0.12 180)',  // iOS Mint
        'oklch(0.6 0.1 220)',    // iOS Cyan
        'oklch(0.5 0.08 280)',   // iOS Indigo
        'oklch(0.65 0.1 40)',    // iOS Amber
    ];
    let hash = 0;
    for (let i = 0; i < pid.length; i++) {
        hash = pid.charCodeAt(i) + ((hash << 5) - hash);
    }
    return colors[Math.abs(hash) % colors.length];
};

// 格式化时间（秒转为可读格式）
const formatDuration = (seconds: number | undefined): string => {
    if (seconds === undefined || seconds < 0) return '--';

    const days = Math.floor(seconds / (24 * 3600));
    const hours = Math.floor((seconds % (24 * 3600)) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
};

// 格式化预计结束时间
const formatEstimatedEndTime = (remainingSeconds: number | undefined): string => {
    if (remainingSeconds === undefined || remainingSeconds < 0) return '--';

    const endTime = new Date(Date.now() + remainingSeconds * 1000);
    return endTime.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    });
};

// 将时间字符串转为秒数
// 支持格式: "0:01:14" (MM:SS), "15:30:52" (HH:MM:SS), 或纯秒数 "3600"
const parseTimeToSeconds = (timeStr: string | undefined): number | undefined => {
    if (!timeStr || timeStr.trim() === '') return undefined;

    const trimmed = timeStr.trim();

    // 如果是纯数字，直接返回
    if (/^\d+$/.test(trimmed)) {
        return parseInt(trimmed, 10);
    }

    // 解析 HH:MM:SS 或 MM:SS
    const parts = trimmed.split(':').map(Number);
    if (parts.length === 3) {
        // HH:MM:SS
        const [hours, minutes, seconds] = parts;
        if (!isNaN(hours) && !isNaN(minutes) && !isNaN(seconds)) {
            return hours * 3600 + minutes * 60 + seconds;
        }
    } else if (parts.length === 2) {
        // MM:SS
        const [minutes, seconds] = parts;
        if (!isNaN(minutes) && !isNaN(seconds)) {
            return minutes * 60 + seconds;
        }
    }

    return undefined;
};

const formatQueryTime = (isoTime?: string): string => {
    if (!isoTime) return '--';
    const date = new Date(isoTime);
    if (Number.isNaN(date.getTime())) return '--';
    return date.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
};

// 计算指数移动平均（EMA）平滑曲线（只平滑 loss，保留 lr 和 step）
const calculateEMA = (
    data: { step: number; loss: number; lr: number }[],
    smoothingFactor: number
): { step: number; loss: number; lr: number }[] => {
    if (smoothingFactor <= 0 || data.length === 0) return data;
    
    const alpha = 1 - smoothingFactor;
    const result: { step: number; loss: number; lr: number }[] = [];
    let smoothedLoss = data[0].loss;
    
    result.push({ step: data[0].step, loss: smoothedLoss, lr: data[0].lr });
    
    for (let i = 1; i < data.length; i++) {
        smoothedLoss = alpha * data[i].loss + (1 - alpha) * smoothedLoss;
        result.push({ step: data[i].step, loss: smoothedLoss, lr: data[i].lr });
    }
    
    return result;
};

const toFiniteNumber = (value: unknown): number | undefined => {
    if (value === null || value === undefined || value === '') return undefined;
    const numberValue = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(numberValue) ? numberValue : undefined;
};

const parseMetricsJSONPayload = (text: string): any | null => {
    const trimmed = text.trim();
    const candidates = [trimmed];

    const firstBrace = trimmed.indexOf('{');
    const lastBrace = trimmed.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
        candidates.push(trimmed.slice(firstBrace, lastBrace + 1));
    }

    for (const candidate of candidates) {
        try {
            return JSON.parse(candidate);
        } catch {
            // Try the next possible JSON slice.
        }
    }

    return null;
};

/**
 * Parse metrics from JSON format
 * Supports the new JSON structure with metrics.pid, metrics.history, etc.
 */
const parseMetricsFromJSON = (text: string): Map<string, MetricPoint[]> => {
    const result = new Map<string, MetricPoint[]>();

    if (!text || text.trim() === '') {
        return result;
    }

    try {
        const jsonData = parseMetricsJSONPayload(text);
        if (!jsonData) {
            return result;
        }

        // Check if it has metrics structure
        if (jsonData.metrics) {
            const metrics = jsonData.metrics;
            const pid = metrics.pid;

            if (!pid) {
                return result;
            }

            const metricPoints: MetricPoint[] = [];

            // 从 metrics 级别获取全局时间和步数信息
            const globalElapsedTime = parseTimeToSeconds(metrics.elapsed_time);
            const globalRemainingTime = parseTimeToSeconds(metrics.remaining_time);
            const globalTotalSteps = metrics.total_steps;
            const globalLearningRate = toFiniteNumber(metrics.latest_learning_rate);

            // 1. Extract from history array
            if (metrics.history && Array.isArray(metrics.history)) {
                metrics.history.forEach((item: any) => {
                    const step = toFiniteNumber(item.step ?? item._step);
                    const loss = toFiniteNumber(item.loss);
                    if (step !== undefined && loss !== undefined) {
                        const lr = toFiniteNumber(item.lr ?? item.learning_rate) ?? globalLearningRate ?? 0;
                        metricPoints.push({
                            step,
                            loss,
                            lr,
                            elapsedTime: globalElapsedTime,
                            remainingTime: globalRemainingTime,
                            totalSteps: globalTotalSteps
                        });
                    }
                });
            }

            // 2. Extract latest info if no history or to supplement
            const latestStep = toFiniteNumber(metrics.latest_step);
            const latestLoss = toFiniteNumber(metrics.latest_loss);
            if (latestStep !== undefined && latestLoss !== undefined) {
                const latestPoint: MetricPoint = {
                    step: latestStep,
                    loss: latestLoss,
                    lr: globalLearningRate ?? 0,
                    elapsedTime: parseTimeToSeconds(metrics.elapsed_time),
                    remainingTime: parseTimeToSeconds(metrics.remaining_time),
                    totalSteps: metrics.total_steps
                };

                // Add latest point if no history or if latest is newer than history
                if (metricPoints.length === 0) {
                    metricPoints.push(latestPoint);
                } else {
                    const lastHistoryPoint = metricPoints[metricPoints.length - 1];
                    if (latestPoint.step > lastHistoryPoint.step) {
                        metricPoints.push(latestPoint);
                    }
                }
            }

            if (metricPoints.length > 0) {
                result.set(pid, metricPoints);
            }
        }
    } catch (e) {
        // Not valid JSON, return empty result
    }

    return result;
};

const parseStepLossLrPair = (text: string): MetricPoint[] => {
    const results: MetricPoint[] = [];
    // 支持格式: stepxxx,lossxxx,lrxxx,elapsed_timexxx,remaining_timexxx,total_stepsxxx
    // 时间格式支持: 纯秒数(3600) 或 HH:MM:SS(0:01:14) 或 MM:SS(1:14)
    // 或者简化格式: stepxxx,lossxxx,lrxxx
    const pattern = /step\s*(\d+)\s*,\s*loss\s*([\d.eE+-]+|nan|NaN)\s*,\s*lr\s*([\d.eE+-]+)(?:\s*,\s*elapsed_time\s*([\d:]+))?(?:\s*,\s*remaining_time\s*([\d:]+))?(?:\s*,\s*total_steps\s*(\d+))?/gi;

    let match;
    while ((match = pattern.exec(text)) !== null) {
        const step = parseInt(match[1], 10);
        const loss = parseFloat(match[2]);
        const lr = parseFloat(match[3]);
        const elapsedTimeStr = match[4];
        const remainingTimeStr = match[5];
        const totalSteps = match[6] ? parseInt(match[6], 10) : undefined;

        // 解析时间字符串（支持 HH:MM:SS 或纯秒数）
        const elapsedTime = parseTimeToSeconds(elapsedTimeStr);
        const remainingTime = parseTimeToSeconds(remainingTimeStr);

        // 分别验证各个字段，允许 loss 为 NaN（用于异常检测）
        const isValidStep = !isNaN(step) && step >= 0;
        const isValidLr = !isNaN(lr) && lr >= 0;

        if (isValidStep && isValidLr) {
            // loss 可以是正常数值或 NaN（NaN 会被异常检测捕获）
            results.push({ step, loss, lr, elapsedTime, remainingTime, totalSteps });
        }
    }

    return results;
};

const parseMultiProcessMetrics = (text: string): Map<string, MetricPoint[]> => {
    const result = new Map<string, MetricPoint[]>();

    if (!text || text.trim() === '') {
        return result;
    }

    // First try to parse as JSON
    const jsonResult = parseMetricsFromJSON(text);
    if (jsonResult.size > 0) {
        return jsonResult;
    }

    // Fallback to text parsing

    // 简化的正则：匹配 "进程数字," 开头，然后捕获到下一个"进程"或结尾
    // 支持前缀如 "测试训练##进程1456"
    const processPattern = /进程(\d+),([^]*?)(?=进程\d+|$)/gi;

    let match;
    while ((match = processPattern.exec(text)) !== null) {
        const pid = match[1];
        const dataStr = match[2].trim();

        if (dataStr) {
            const metrics = parseStepLossLrPair(dataStr);
            if (metrics.length > 0) {
                result.set(pid, metrics);
            }
        }
    }

    return result;
};

const cloneProcessDataMap = (
    source: Map<string, ProcessData> | undefined,
    smoothingFactor: number,
): Map<string, ProcessData> => {
    const result = new Map<string, ProcessData>();
    source?.forEach((process, pid) => {
        const metrics = process.metrics.map((metric) => ({ ...metric }));
        result.set(pid, {
            ...process,
            metrics,
            smoothedMetrics: calculateEMA(metrics, smoothingFactor),
        });
    });
    return result;
};

const TrainingMetricsPanel = ({
    onClose,
    replies,
    cacheKey,
    initialCache,
    onCacheChange,
    onAskAI,
    onMonitorTraining,
    monitorStatus,
    isInputDisabled,
}: Props) => {
    const { t } = useTranslation();
    const { messageApi } = useMessageApi();
    const chartRef = useRef<HTMLDivElement>(null);
    const [isExporting, setIsExporting] = useState(false);
    const [showExportMenu, setShowExportMenu] = useState(false);
    const [isPidListCollapsed, setIsPidListCollapsed] = useState(false);
    const exportMenuRef = useRef<HTMLDivElement>(null);
    type ExportFormat = 'png' | 'csv';
    
    // 询问AI相关状态
    const [isAskAIDialogOpen, setIsAskAIDialogOpen] = useState(false);
    const [isAskingAI, setIsAskingAI] = useState(false);
    const [customQuestion, setCustomQuestion] = useState('');

    // 点击外部关闭下拉菜单
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (exportMenuRef.current && !exportMenuRef.current.contains(event.target as Node)) {
                setShowExportMenu(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);
    const [processDataMap, setProcessDataMap] = useState<Map<string, ProcessData>>(
        () => cloneProcessDataMap(initialCache?.processDataMap, 0),
    );
    const [selectedPid, setSelectedPid] = useState<string | null>(
        () => initialCache?.selectedPid ?? null,
    );
    const [isComparisonMode, setIsComparisonMode] = useState(false);
    const [selectedPids, setSelectedPids] = useState<Set<string>>(new Set());
    const [smoothingFactor, setSmoothingFactor] = useState(0); // 平滑度：0-0.999，默认0
    const [showSmoothed, setShowSmoothed] = useState(true); // 是否显示平滑曲线
    const [showRaw, setShowRaw] = useState(true); // 是否显示原始数据
    const [isLogScaleLoss, setIsLogScaleLoss] = useState(false); // Loss图表是否使用对数刻度
    const [isLogScaleLR, setIsLogScaleLR] = useState(false); // LR图表是否使用对数刻度
    
    // 缩放状态：记录当前显示的 step 范围
    const [zoomDomain, setZoomDomain] = useState<{
        loss: { min: number; max: number } | null;
        lr: { min: number; max: number } | null;
    }>({ loss: null, lr: null });
    
    // 全屏状态
    const [fullscreenChart, setFullscreenChart] = useState<'loss' | 'lr' | null>(null);
    
    // 拖拽状态
    const [isDragging, setIsDragging] = useState<{ loss: boolean; lr: boolean }>({ loss: false, lr: false });
    const dragStartRef = useRef<{ x: number; domain: { min: number; max: number } | null } | null>(null);
    
    // Loss 异常警告状态
    const [dismissedWarnings, setDismissedWarnings] = useState<Set<string>>(new Set());
    const [activeWarning, setActiveWarning] = useState<{
        runId: string;
        type: 'nan' | 'high';
        value: number;
        suggestions: string[];
    } | null>(null);
    const cacheChangeRef = useRef(onCacheChange);
    const cacheKeyRef = useRef(cacheKey);
    cacheChangeRef.current = onCacheChange;

    useEffect(() => {
        if (cacheKeyRef.current === cacheKey) {
            return;
        }
        cacheKeyRef.current = cacheKey;
        if (!initialCache) {
            setProcessDataMap(new Map());
            setSelectedPid(null);
            return;
        }
        setProcessDataMap(cloneProcessDataMap(initialCache.processDataMap, smoothingFactor));
        setSelectedPid(initialCache.selectedPid);
    }, [cacheKey, initialCache, smoothingFactor]);

    useEffect(() => {
        cacheChangeRef.current?.({
            processDataMap: cloneProcessDataMap(processDataMap, smoothingFactor),
            selectedPid,
        });
    }, [processDataMap, selectedPid, smoothingFactor]);

    const lastMonitorQueryTime = monitorStatus?.lastResultAt || monitorStatus?.lastQueryAt;
    const isWaitingForMetricWrite =
        !monitorStatus?.hasMetrics &&
        monitorStatus?.message === '训练已启动，等待指标写入';
    const monitorStatusText = (() => {
        if (monitorStatus?.isQuerying) {
            return t('training.monitor-querying', { defaultValue: '查询中...' }) as string;
        }
        if (!monitorStatus?.lastResultAt) {
            return t('training.monitor-waiting', { defaultValue: '等待查询' }) as string;
        }
        if (!monitorStatus.hasMetrics) {
            if (isWaitingForMetricWrite) {
                return t('training.monitor-waiting-for-metrics', { defaultValue: '等待指标写入' }) as string;
            }
            return t('training.monitor-no-metrics', { defaultValue: '尚未获取到指标' }) as string;
        }
        return monitorStatus.hasNewData
            ? (t('training.monitor-metrics-ready', { defaultValue: '已获取指标' }) as string)
            : (t('training.monitor-no-new-metrics', { defaultValue: '暂无新指标' }) as string);
    })();
    
    const MAX_COMPARISON_COUNT = 5;

    useEffect(() => {
        const parsedByPid = new Map<string, MetricPoint[]>();

        replies.forEach((reply) => {
            reply.messages.forEach((msg) => {
                // 检查是否是 monitor 消息或包含 tool_result 的消息
                const isMonitorMessage = msg.name?.toLowerCase().includes('monitor');
                const hasToolResult = Array.isArray(msg.content) && msg.content.some((b: { type?: string }) => b.type === 'tool_result');

                if (isMonitorMessage || hasToolResult) {
                    const text = extractTextFromContent(msg.content);
                    const parsed = parseMultiProcessMetrics(text);

                    parsed.forEach((metrics, pid) => {
                        if (metrics.length === 0) {
                            return;
                        }

                        const existingMetrics = parsedByPid.get(pid) || [];
                        parsedByPid.set(pid, [...existingMetrics, ...metrics]);
                    });
                }
            });
        });

        if (parsedByPid.size === 0) {
            return;
        }

        setProcessDataMap((previousDataMap) => {
            const newDataMap = new Map(previousDataMap);

            parsedByPid.forEach((metrics, pid) => {
                const existingData = newDataMap.get(pid);
                            
                if (existingData) {
                    const metricsByStep = new Map(existingData.metrics.map(m => [m.step, m]));
                    metrics.forEach((metric) => {
                        const existingMetric = metricsByStep.get(metric.step);
                        metricsByStep.set(metric.step, {
                            ...(existingMetric ?? {}),
                            ...metric,
                            elapsedTime: metric.elapsedTime ?? existingMetric?.elapsedTime,
                            remainingTime: metric.remainingTime ?? existingMetric?.remainingTime,
                            totalSteps: metric.totalSteps ?? existingMetric?.totalSteps,
                        });
                    });

                    const updatedMetrics = Array.from(metricsByStep.values())
                        .sort((a, b) => a.step - b.step);

                    // 计算平滑数据（只针对当前选中的PID）
                    const updatedSmoothed = selectedPid === pid
                        ? calculateEMA(updatedMetrics, smoothingFactor)
                        : existingData.smoothedMetrics;

                    const latest = updatedMetrics[updatedMetrics.length - 1];
                    const totalSteps = latest.totalSteps || existingData.totalSteps;
                    const progress = totalSteps && totalSteps > 0
                        ? Math.round((latest.step / totalSteps) * 100)
                        : existingData.progress;

                    newDataMap.set(pid, {
                        ...existingData,
                        metrics: updatedMetrics,
                        smoothedMetrics: updatedSmoothed,
                        latestStep: latest.step,
                        latestLoss: latest.loss,
                        latestLR: latest.lr,
                        latestElapsedTime: latest.elapsedTime ?? existingData.latestElapsedTime,
                        latestRemainingTime: latest.remainingTime ?? existingData.latestRemainingTime,
                        totalSteps: totalSteps,
                        progress: progress,
                    });
                    return;
                }

                const sortedMetrics = metrics.sort((a, b) => a.step - b.step);
                const latest = sortedMetrics[sortedMetrics.length - 1];
                const totalSteps = latest.totalSteps;
                const progress = totalSteps && totalSteps > 0
                    ? Math.round((latest.step / totalSteps) * 100)
                    : undefined;

                // 新PID只计算基础平滑数据（平滑度为0，即原始数据）
                newDataMap.set(pid, {
                    pid,
                    metrics: sortedMetrics,
                    smoothedMetrics: sortedMetrics,
                    color: generateColor(pid),
                    latestStep: latest.step,
                    latestLoss: latest.loss,
                    latestLR: latest.lr,
                    latestElapsedTime: latest.elapsedTime,
                    latestRemainingTime: latest.remainingTime,
                    totalSteps: totalSteps,
                    progress: progress,
                });
            });

            return newDataMap;
        });

        if (!selectedPid) {
            const firstPid = Array.from(parsedByPid.keys())[0];
            setSelectedPid(firstPid);
        }
    }, [replies, selectedPid, smoothingFactor]);

    // 当平滑度变化时，重新计算所有相关PID的平滑数据
    useEffect(() => {
        const pidsToUpdate = isComparisonMode
            ? Array.from(selectedPids)  // 对比模式：更新所有选中的PID
            : selectedPid ? [selectedPid] : [];  // 单选模式：更新当前选中的PID

        if (pidsToUpdate.length === 0) return;

        const newDataMap = new Map(processDataMap);

        pidsToUpdate.forEach(pid => {
            const process = newDataMap.get(pid);
            if (process) {
                const smoothedMetrics = calculateEMA(process.metrics, smoothingFactor);
                newDataMap.set(pid, {
                    ...process,
                    smoothedMetrics,
                });
            }
        });

        setProcessDataMap(newDataMap);
    }, [smoothingFactor, selectedPid, selectedPids, isComparisonMode]);

    const selectedProcess = selectedPid ? processDataMap.get(selectedPid) : null;

    const toggleComparisonMode = useCallback((enabled: boolean) => {
        setIsComparisonMode(enabled);
        if (enabled && selectedPid) {
            setSelectedPids(new Set([selectedPid]));
        } else if (!enabled && selectedPids.size > 0) {
            const lastPid = Array.from(selectedPids).pop();
            setSelectedPid(lastPid || null);
            setSelectedPids(new Set());
        }
    }, [selectedPid, selectedPids]);

    const togglePidSelection = useCallback((pid: string, checked: boolean) => {
        setSelectedPids(prev => {
            const newSet = new Set(prev);
            if (checked) {
                if (newSet.size >= MAX_COMPARISON_COUNT) {
                    messageApi.warning(t('training.max-comparison'));
                    return prev;
                }
                newSet.add(pid);
            } else {
                newSet.delete(pid);
                if (newSet.size === 0) {
                    return prev;
                }
            }
            return newSet;
        });
    }, [messageApi, t]);

    const mergedComparisonData = useMemo(() => {
        if (!isComparisonMode || selectedPids.size === 0) return [];

        const stepSet = new Set<number>();
        selectedPids.forEach(pid => {
            const process = processDataMap.get(pid);
            process?.metrics.forEach(m => stepSet.add(m.step));
        });

        return Array.from(stepSet).sort((a, b) => a - b).map(step => {
            const dataPoint: any = { step };
            selectedPids.forEach(pid => {
                const process = processDataMap.get(pid);
                // 原始数据
                const rawMetric = process?.metrics.find(m => m.step === step);
                dataPoint[`raw${pid}`] = rawMetric?.loss ?? null;
                // 平滑数据
                const smoothedMetric = process?.smoothedMetrics.find(m => m.step === step);
                dataPoint[`smooth${pid}`] = smoothedMetric?.loss ?? null;
            });
            return dataPoint;
        });
    }, [isComparisonMode, selectedPids, processDataMap]);

    // 导出 PNG
    const exportPNG = useCallback(async () => {
        if (!chartRef.current || !selectedProcess) {
            messageApi.error('Chart element not found');
            return;
        }
        
        setIsExporting(true);
        
        try {
            const canvas = await html2canvas(chartRef.current, {
                backgroundColor: '#ffffff',
                scale: 2,
                logging: false,
                useCORS: true,
                allowTaint: true,
            });
            
            const link = document.createElement('a');
            // 文件名根据模式区分
            const fileName = isComparisonMode
                ? `training-metrics-multi-${Date.now()}.png`
                : `training-metrics-${selectedProcess.pid}-${Date.now()}.png`;
            link.download = fileName;
            link.href = canvas.toDataURL('image/png');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            messageApi.success(t('training.export-success'));
        } catch (error) {
            console.error('Export PNG failed:', error);
            messageApi.error(
                t('training.export-failed') || 
                `Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
        } finally {
            setIsExporting(false);
        }
    }, [messageApi, t, selectedProcess]);

    // 导出 CSV
    const exportCSV = useCallback(() => {
        const pidsToExport = isComparisonMode
            ? Array.from(selectedPids)  // 对比模式：导出所有选中的 PID
            : selectedPid ? [selectedPid] : [];  // 单选模式：导出当前选中的 PID
        
        if (pidsToExport.length === 0) return;
        
        const headers = ['PID', 'Step', 'Loss', 'SmoothedLoss', 'SmoothingFactor'];
        const smoothingPercent = Math.round(smoothingFactor * 100);
        
        // 收集所有选中的 PID 数据
        const allRows: (string | number)[][] = [];
        pidsToExport.forEach(pid => {
            const process = processDataMap.get(pid);
            if (process) {
                process.metrics.forEach((m, i) => {
                    allRows.push([
                        process.pid,
                        m.step,
                        m.loss.toFixed(6),
                        (process.smoothedMetrics[i]?.loss ?? m.loss).toFixed(6),
                        `${smoothingPercent}%`,
                    ]);
                });
            }
        });
        
        const csvContent = [
            headers.join(','),
            ...allRows.map(row => row.join(','))
        ].join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        
        // 文件名根据模式区分
        const fileName = isComparisonMode
            ? `training-metrics-multi-${Date.now()}.csv`
            : `training-metrics-${selectedPid}-${Date.now()}.csv`;
        link.download = fileName;
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        messageApi.success(t('training.export-csv-success') || 'CSV exported successfully');
    }, [isComparisonMode, selectedPids, selectedPid, processDataMap, smoothingFactor, messageApi, t]);

    // 统一导出入口
    const handleExport = useCallback((format: ExportFormat) => {
        if (format === 'png') {
            exportPNG();
        } else {
            exportCSV();
        }
        setShowExportMenu(false);
    }, [exportPNG, exportCSV]);

    // 询问AI：捕获图表并发送给AI分析
    const handleAskAI = useCallback(async () => {
        if (!selectedProcess) {
            messageApi.error(t('no-process') || 'No process selected');
            return;
        }

        if (!onAskAI) {
            messageApi.error(t('ask-ai-not-available') || 'AI analysis is not available');
            return;
        }

        // 等待 DOM 渲染完成
        await new Promise(resolve => setTimeout(resolve, 100));

        if (!chartRef.current) {
            messageApi.error(t('no-chart') || 'No chart to capture');
            return;
        }

        setIsAskingAI(true);

        try {
            // 1. 截图
            const canvas = await html2canvas(chartRef.current, {
                backgroundColor: '#ffffff',
                scale: 2,
                logging: false,
                useCORS: true,
                allowTaint: true,
            });

            // 2. 获取Base64数据（去掉data:image/png;base64前缀）
            const base64Data = canvas.toDataURL('image/png').split(',')[1];

            // 3. 构建提示文本
            const modeText = isComparisonMode 
                ? (t('training.comparison-mode') || '对比模式')
                : (t('single-process') || '单进程');
            
            const defaultPrompt = t('ask-ai-default-prompt') || 
                '请帮我分析这个训练曲线的趋势和是否存在异常。';
            
            const userQuestion = customQuestion.trim() || defaultPrompt;

            const promptText = `[${t('curve-analysis') || '训练曲线分析'}] ${modeText} - PID ${selectedProcess.pid}\n\n` +
                `${userQuestion}\n\n` +
                `${t('current-info') || '当前训练信息'}：\n` +
                `- ${t('training.process') || '进程'}: PID ${selectedProcess.pid}\n` +
                `- ${t('training.step') || '当前步数'}: ${selectedProcess.latestStep}\n` +
                `- ${t('training.loss') || '当前Loss'}: ${selectedProcess.latestLoss?.toFixed(6) ?? '--'}\n` +
                `- ${t('training.lr') || '当前LR'}: ${selectedProcess.latestLR && selectedProcess.latestLR > 0 ? selectedProcess.latestLR.toExponential(2) : 'warmup'}`;

            // 4. 构建ContentBlocks
            const blocks: ContentBlocks = [
                {
                    type: BlockType.TEXT,
                    text: promptText,
                } as { type: BlockType.TEXT; text: string },
                {
                    type: BlockType.IMAGE,
                    source: {
                        type: SourceType.BASE64,
                        media_type: 'image/png',
                        data: base64Data,
                    },
                } as { type: BlockType.IMAGE; source: { type: SourceType.BASE64; media_type: string; data: string } },
            ];

            // 5. 发送
            onAskAI(blocks);

            messageApi.success(t('ask-ai-success') || '已发送给AI分析');
            setIsAskAIDialogOpen(false);
            setCustomQuestion('');
        } catch (error) {
            console.error('Ask AI failed:', error);
            messageApi.error(
                t('ask-ai-failed') || 
                `发送失败: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
        } finally {
            setIsAskingAI(false);
        }
    }, [selectedProcess, isComparisonMode, selectedPids, smoothingFactor, isLogScaleLoss, customQuestion, onAskAI, messageApi, t]);

    // 计算可见数据的 Y 轴范围
    const calculateYDomain = useCallback((
        data: { step: number; loss?: number; lr?: number }[],
        xDomain: { min: number; max: number } | null,
        key: 'loss' | 'lr',
        isLogScale: boolean
    ): [number, number] => {
        if (data.length === 0) return isLogScale ? [0.001, 1] : [0, 1];
        
        // 过滤可见范围的数据
        const visibleData = xDomain 
            ? data.filter(d => d.step >= xDomain.min && d.step <= xDomain.max)
            : data;
        
        if (visibleData.length === 0) return isLogScale ? [0.001, 1] : [0, 1];
        
        const values = visibleData.map(d => key === 'loss' ? d.loss : d.lr).filter((v): v is number => v !== undefined && !isNaN(v));
        
        if (values.length === 0) return isLogScale ? [0.001, 1] : [0, 1];
        
        let min = Math.min(...values);
        let max = Math.max(...values);
        
        // 添加一些边距
        const padding = (max - min) * 0.1;
        min = Math.max(0, min - padding);
        max = max + padding;
        
        if (isLogScale) {
            // 对数刻度需要正值
            min = Math.max(0.0001, min);
            max = Math.max(min * 1.1, max);
        }
        
        return [min, max];
    }, []);

    // 处理 Alt+Scroll 缩放
    const handleWheel = useCallback((e: React.WheelEvent, chartType: 'loss' | 'lr') => {
        if (!e.altKey) return; // 必须按住 Alt 键
        
        e.preventDefault();
        e.stopPropagation();
        
        const process = selectedProcess;
        if (!process) return;
        
        const data = chartType === 'loss' 
            ? process.metrics 
            : process.metrics.map(m => ({ step: m.step, lr: m.lr }));
        
        if (data.length < 2) return;
        
        const currentDomain = zoomDomain[chartType];
        const minStep = data[0].step;
        const maxStep = data[data.length - 1].step;
        
        let currentMin = currentDomain?.min ?? minStep;
        let currentMax = currentDomain?.max ?? maxStep;
        
        const range = currentMax - currentMin;
        const center = (currentMin + currentMax) / 2;
        
        // 缩放因子：每次滚动缩放 15%
        const zoomFactor = e.deltaY > 0 ? 1.15 : 0.85;
        const newRange = Math.max(10, range * zoomFactor); // 最小范围 10 steps
        
        let newMin = center - newRange / 2;
        let newMax = center + newRange / 2;
        
        // 限制在数据范围内
        if (newMin < minStep) {
            newMax += minStep - newMin;
            newMin = minStep;
        }
        if (newMax > maxStep) {
            newMin -= newMax - maxStep;
            newMax = maxStep;
        }
        
        // 确保最小范围
        if (newMax - newMin < 10) {
            const mid = (newMin + newMax) / 2;
            newMin = mid - 5;
            newMax = mid + 5;
        }
        
        setZoomDomain(prev => ({
            ...prev,
            [chartType]: { min: newMin, max: newMax }
        }));
    }, [selectedProcess, zoomDomain]);

    // Fit domain to data
    const handleFitDomain = useCallback((chartType: 'loss' | 'lr') => {
        setZoomDomain(prev => ({
            ...prev,
            [chartType]: null // null 表示显示全部数据
        }));
    }, []);

    // 处理全屏
    const toggleFullscreen = useCallback((chartType: 'loss' | 'lr') => {
        setFullscreenChart(prev => prev === chartType ? null : chartType);
    }, []);

    // 处理鼠标按下 - 开始拖拽
    const handleMouseDown = useCallback((e: React.MouseEvent, chartType: 'loss' | 'lr') => {
        setIsDragging(prev => ({ ...prev, [chartType]: true }));
        dragStartRef.current = {
            x: e.clientX,
            domain: zoomDomain[chartType]
        };
    }, [zoomDomain]);

    // 处理鼠标移动 - 平移视图
    const handleMouseMove = useCallback((e: React.MouseEvent, chartType: 'loss' | 'lr') => {
        if (!isDragging[chartType] || !dragStartRef.current || !selectedProcess) return;
        
        if (selectedProcess.metrics.length < 2) return;
        
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const chartWidth = rect.width - 70; // 减去边距
        const deltaX = e.clientX - dragStartRef.current.x;
        
        const startDomain = dragStartRef.current.domain;
        const minStep = selectedProcess.metrics[0].step;
        const maxStep = selectedProcess.metrics[selectedProcess.metrics.length - 1].step;
        
        // 计算当前显示的 step 范围
        const currentMin = startDomain?.min ?? minStep;
        const currentMax = startDomain?.max ?? maxStep;
        const currentRange = currentMax - currentMin;
        
        // 根据拖拽距离计算新的范围
        const stepPerPixel = currentRange / chartWidth;
        const stepDelta = deltaX * stepPerPixel;
        
        let newMin = currentMin - stepDelta;
        let newMax = currentMax - stepDelta;
        
        // 边界检查
        if (newMin < minStep) {
            newMax += minStep - newMin;
            newMin = minStep;
        }
        if (newMax > maxStep) {
            newMin -= newMax - maxStep;
            newMax = maxStep;
        }
        
        setZoomDomain(prev => ({
            ...prev,
            [chartType]: { min: newMin, max: newMax }
        }));
    }, [isDragging, selectedProcess, setZoomDomain]);

    // 处理鼠标释放 - 结束拖拽
    const handleMouseUp = useCallback((chartType: 'loss' | 'lr') => {
        setIsDragging(prev => ({ ...prev, [chartType]: false }));
        dragStartRef.current = null;
    }, []);

    // 处理鼠标离开 - 结束拖拽
    const handleMouseLeave = useCallback((chartType: 'loss' | 'lr') => {
        if (isDragging[chartType]) {
            setIsDragging(prev => ({ ...prev, [chartType]: false }));
            dragStartRef.current = null;
        }
    }, [isDragging]);

    // 全屏时监听 ESC 键
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && fullscreenChart) {
                setFullscreenChart(null);
            }
        };
        
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [fullscreenChart]);

    // Loss 异常检测
    useEffect(() => {
        if (!selectedProcess) return;
        
        // 检查是否已经提示过
        if (dismissedWarnings.has(selectedProcess.pid)) return;
        
        // 检查是否有任何异常 Loss 值（任何数据点）
        const abnormalMetrics = selectedProcess.metrics.filter(m => {
            const loss = m.loss;
            return isNaN(loss) || loss > 100;
        });
        
        if (abnormalMetrics.length === 0) return;
        
        // 取最新的异常数据点
        const latestAbnormal = abnormalMetrics[abnormalMetrics.length - 1];
        const loss = latestAbnormal.loss;
        
        let warning = null;
        
        if (isNaN(loss)) {
            warning = {
                runId: selectedProcess.pid,
                type: 'nan' as const,
                value: loss,
                suggestions: [
                    t('training.loss-suggestion-nan-1') || '学习率可能过高，建议降低学习率（如除以10）',
                    t('training.loss-suggestion-nan-2') || '检查数据是否存在问题（如标签错误、数据泄露）',
                    t('training.loss-suggestion-nan-3') || '尝试使用梯度裁剪（gradient clipping）',
                    t('training.loss-suggestion-nan-4') || '检查损失函数实现是否正确'
                ]
            };
        } else if (loss > 100) {
            warning = {
                runId: selectedProcess.pid,
                type: 'high' as const,
                value: loss,
                suggestions: [
                    t('training.loss-suggestion-high-1') || '学习率可能过高，建议降低学习率',
                    t('training.loss-suggestion-high-2') || '检查模型初始化是否正常',
                    t('training.loss-suggestion-high-3') || '尝试使用学习率预热（warmup）策略',
                    t('training.loss-suggestion-high-4') || '检查输入数据是否经过正确的归一化'
                ]
            };
        }
        
        if (warning) {
            setActiveWarning(warning);
        }
    }, [selectedProcess, dismissedWarnings, t]);

    const renderPidListItem = (process: ProcessData) => {
        if (isComparisonMode) {
            return (
                <label
                    key={process.pid}
                    className={`w-full p-2 flex items-start gap-2 cursor-pointer border-b border-border transition-colors hover:bg-muted ${
                        selectedPids.has(process.pid) ? 'bg-primary/5' : ''
                    }`}
                >
                    <input
                        type="checkbox"
                        checked={selectedPids.has(process.pid)}
                        onChange={(e) => togglePidSelection(process.pid, e.target.checked)}
                        className="w-4 h-4 mt-0.5 rounded border-border flex-shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1">
                            <div
                                className="w-2 h-2 rounded-full flex-shrink-0"
                                style={{ backgroundColor: process.color }}
                            />
                            <span className="text-xs font-medium truncate">PID {process.pid}</span>
                        </div>
                        <div className="text-xs text-muted-foreground">
                            Step: {process.latestStep}
                        </div>
                    </div>
                </label>
            );
        }
        
        return (
            <button
                key={process.pid}
                onClick={() => setSelectedPid(process.pid)}
                className={`w-full p-2 text-left text-sm border-b border-border transition-colors ${
                    selectedPid === process.pid
                        ? 'bg-primary/10 border-l-4 border-l-primary'
                        : 'hover:bg-muted border-l-4 border-l-transparent'
                }`}
            >
                <div className="flex items-center gap-2">
                    <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: process.color }}
                    />
                    <span className="font-medium">PID {process.pid}</span>
                </div>
                <div className="mt-1 text-xs text-muted-foreground pl-5">
                    Step: {process.latestStep}
                </div>
            </button>
        );
    };

    return (
        <div className="w-full bg-background flex flex-col h-full overflow-hidden">
            <div className="h-12 border-b border-border flex items-center justify-between px-3">
                <div className="min-w-0 flex items-center gap-2">
                    <h3 className="font-semibold text-sm">{t('training.metrics') || 'Training'}</h3>
                    <div
                        className="hidden min-w-0 items-center gap-1.5 rounded-full border border-border bg-muted/40 px-2 py-1 text-[11px] text-muted-foreground sm:flex"
                        title={`${monitorStatusText} · ${t('training.last-query', { defaultValue: '上次查询' })}: ${formatQueryTime(lastMonitorQueryTime)}`}
                    >
                        {monitorStatus?.isQuerying ? (
                            <Loader2 className="h-3 w-3 animate-spin text-primary" />
                        ) : (
                            <Clock3 className="h-3 w-3" />
                        )}
                        <span className={monitorStatus?.isQuerying ? 'text-primary' : ''}>
                            {monitorStatusText}
                        </span>
                        <span className="max-w-[84px] truncate">
                            {formatQueryTime(lastMonitorQueryTime)}
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {/* 平滑度控制 */}
                    {selectedProcess && (
                        <div className="flex items-center gap-2" title={t('training.smoothing') || 'Smoothing'}>
                            <svg className="h-3 w-3 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            <input
                                type="range"
                                min="0"
                                max="1000"
                                value={Math.round(smoothingFactor * 1000)}
                                onChange={(e) => setSmoothingFactor(parseInt(e.target.value) / 1000)}
                                className="w-20 h-1.5 bg-muted rounded-lg appearance-none cursor-pointer"
                            />
                            <span className="text-xs w-10 text-right">{(smoothingFactor * 100).toFixed(1)}%</span>
                        </div>
                    )}
                    <div className="flex items-center gap-1">
                        {/* 导出下拉菜单 */}
                        <div className="relative" ref={exportMenuRef}>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8"
                                onClick={() => setShowExportMenu(!showExportMenu)}
                                title={t('training.export') || 'Export'}
                                disabled={!selectedProcess || isExporting}
                            >
                                {isExporting ? (
                                    <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                ) : (
                                    <Camera className="h-4 w-4" />
                                )}
                            </Button>
                            
                            {/* 下拉菜单 */}
                            {showExportMenu && (
                                <div className="absolute right-0 top-full mt-1 w-32 bg-background border border-border rounded-md shadow-lg z-50 py-1">
                                    <button
                                        onClick={() => handleExport('png')}
                                        className="w-full px-3 py-2 text-left text-xs hover:bg-muted flex items-center gap-2"
                                    >
                                        <Camera className="h-3 w-3" />
                                        {t('training.export-png') || '导出 PNG'}
                                    </button>
                                    <button
                                        onClick={() => handleExport('csv')}
                                        className="w-full px-3 py-2 text-left text-xs hover:bg-muted flex items-center gap-2"
                                    >
                                        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        {t('training.export-csv') || '导出 CSV'}
                                    </button>
                                </div>
                            )}
                        </div>
                        
                        {/* 询问AI按钮 */}
                        {onAskAI && (
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8"
                                onClick={() => setIsAskAIDialogOpen(true)}
                                title={t('ask-ai') || 'Ask AI'}
                                disabled={!selectedProcess || isAskingAI || isInputDisabled}
                            >
                                {isAskingAI ? (
                                    <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                ) : (
                                    <Sparkles className="h-4 w-4" />
                                )}
                            </Button>
                        )}
                        
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={onClose}
                            title={t('training.close') || 'Close'}
                        >
                            <PanelLeftClose className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </div>

            <div className="flex-1 flex min-h-0 overflow-hidden">
                {/* PID列表侧边栏 - 可折叠 */}
                <div className={`${isPidListCollapsed ? 'w-8' : 'w-32'} border-r border-border flex flex-col transition-all duration-300`}>
                    {/* 折叠/展开按钮 */}
                    <div className="h-8 border-b border-border flex items-center justify-center">
                        <button
                            onClick={() => setIsPidListCollapsed(!isPidListCollapsed)}
                            className="p-1 hover:bg-muted rounded"
                            title={isPidListCollapsed ? '展开列表' : '折叠列表'}
                        >
                            {isPidListCollapsed ? (
                                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            ) : (
                                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                </svg>
                            )}
                        </button>
                    </div>
                    
                    {/* 列表内容 - 折叠时隐藏 */}
                    {!isPidListCollapsed && (
                        <>
                            <div className="p-2 border-b border-border flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    id="comparison-mode"
                                    checked={isComparisonMode}
                                    onChange={(e) => toggleComparisonMode(e.target.checked)}
                                    className="w-4 h-4 rounded border-border"
                                />
                                <label htmlFor="comparison-mode" className="text-xs cursor-pointer">
                                    {t('training.comparison-mode') || '对比'}
                                </label>
                            </div>
                    
                    <div className="p-2 text-xs text-muted-foreground border-b border-border">
                        {t('training.processes') || 'Processes'} ({processDataMap.size})
                    </div>
                    
                    <div className="flex-1 overflow-y-auto">
                        {processDataMap.size > 0 ? (
                            Array.from(processDataMap.values()).map((process) => renderPidListItem(process))
                        ) : (
                            <div className="flex h-full items-center justify-center px-3 text-center">
                                <div className="text-xs leading-5 text-muted-foreground">
                                    {t('training.no-processes', { defaultValue: '暂无训练进程' })}
                                    <div className="mt-1 text-[11px]">
                                        {t('training.monitor-training-hint', { defaultValue: '通过“监控训练”获取训练指标' })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                        </>
                    )}
                </div>

                <div className="flex-1 flex flex-col min-h-0 overflow-auto">
                    {isComparisonMode ? (
                        selectedPids.size > 0 ? (
                            <>
                                <div className="p-3 border-b border-border bg-muted/30">
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="text-xs text-muted-foreground">
                                            {t('training.comparing') || 'Comparing'} {selectedPids.size} {t('training.processes') || 'processes'}
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <label className="flex items-center gap-1 text-xs cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={showRaw}
                                                    onChange={(e) => setShowRaw(e.target.checked)}
                                                    className="w-3.5 h-3.5 rounded border-border"
                                                />
                                                <span>{t('training.raw') || '原始'}</span>
                                            </label>
                                            <label className="flex items-center gap-1 text-xs cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={showSmoothed}
                                                    onChange={(e) => setShowSmoothed(e.target.checked)}
                                                    className="w-3.5 h-3.5 rounded border-border"
                                                />
                                                <span>{t('training.smoothed') || '平滑'}</span>
                                            </label>
                                        </div>
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                        {Array.from(selectedPids).map(pid => {
                                            const process = processDataMap.get(pid);
                                            return process ? (
                                                <div 
                                                    key={pid} 
                                                    className="flex items-center gap-1.5 text-xs bg-background px-2 py-1 rounded border border-border"
                                                >
                                                    <div
                                                        className="w-2.5 h-2.5 rounded-full"
                                                        style={{ backgroundColor: process.color }}
                                                    />
                                                    <span className="font-medium">PID {pid}</span>
                                                    <span className="text-muted-foreground">Step {process.latestStep}</span>
                                                </div>
                                            ) : null;
                                        })}
                                    </div>
                                </div>

                                <div className="flex-1 min-h-0 p-3">
                                    <div className="h-full bg-card rounded-xl border border-border shadow-sm overflow-hidden">
                                        <div ref={chartRef} className="h-full p-2">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <LineChart 
                                                    data={mergedComparisonData} 
                                                    margin={{ top: 10, right: 20, left: 50, bottom: 50 }}
                                                >
                                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                                                    <XAxis
                                                        dataKey="step"
                                                        type="number"
                                                        domain={['dataMin', 'dataMax']}
                                                        tick={{ fontSize: 11, fill: 'var(--foreground)' }}
                                                        tickFormatter={(value) => value.toLocaleString()}
                                                        stroke="var(--border)"
                                                        label={{ value: 'Step', position: 'insideBottom', offset: -35, fontSize: 11, fill: 'var(--muted-foreground)' }}
                                                    />
                                                    <YAxis
                                                        tick={{ fontSize: 11, fill: 'var(--foreground)' }}
                                                        tickFormatter={(value) => value.toFixed(2)}
                                                        width={50}
                                                        stroke="var(--border)"
                                                    />
                                                    <Tooltip
                                                        contentStyle={{
                                                            backgroundColor: 'var(--background)',
                                                            border: '1px solid var(--border)',
                                                            borderRadius: '6px',
                                                            fontSize: '12px',
                                                            color: 'var(--foreground)',
                                                        }}
                                                        labelFormatter={(label) => `Step: ${label}`}
                                                    />
                                                    <Legend
                                                        verticalAlign="bottom"
                                                        height={40}
                                                        wrapperStyle={{ 
                                                            fontSize: '13px',
                                                            paddingTop: '10px',
                                                        }}
                                                        iconType="circle"
                                                        iconSize={10}
                                                    />
                                                    {Array.from(selectedPids).map(pid => {
                                                        const process = processDataMap.get(pid);
                                                        if (!process) return null;

                                                        const lines = [];
                                                        // 原始数据 - 细实线，透明度降低
                                                        if (showRaw) {
                                                            lines.push(
                                                                <Line
                                                                    key={`raw${pid}`}
                                                                    type="monotone"
                                                                    dataKey={`raw${pid}`}
                                                                    name={`PID ${pid}`}
                                                                    stroke={process.color}
                                                                    strokeWidth={1.5}
                                                                    dot={false}
                                                                    isAnimationActive={false}
                                                                    connectNulls
                                                                    opacity={0.7}
                                                                />
                                                            );
                                                        }
                                                        // 平滑数据 - 粗实线（仅当平滑度>0时显示）
                                                        if (showSmoothed && smoothingFactor > 0) {
                                                            lines.push(
                                                                <Line
                                                                    key={`smooth${pid}`}
                                                                    type="monotone"
                                                                    dataKey={`smooth${pid}`}
                                                                    name={`PID ${pid} (平滑)`}
                                                                    stroke={process.color}
                                                                    strokeWidth={2.5}
                                                                    dot={false}
                                                                    isAnimationActive={false}
                                                                    connectNulls
                                                                />
                                                            );
                                                        }
                                                        return lines;
                                                    })}
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <div className="flex-1 flex items-center justify-center p-4">
                                <div className="text-center text-muted-foreground text-sm">
                                    {t('training.select-at-least-one') || 'Please select at least one process'}
                                </div>
                            </div>
                        )
                    ) : (
                        selectedProcess ? (
                            <>
                                {/* 指标卡片区域 */}
                                <div className="p-3 grid grid-cols-2 gap-2 shrink-0">
                                    <div className="grid grid-cols-3 gap-2">
                                        <div className="bg-muted rounded-lg p-2">
                                            <div className="text-xs text-muted-foreground">{t('training.step') || 'Step'}</div>
                                            <div className="text-lg font-semibold">{selectedProcess.latestStep}</div>
                                        </div>
                                        <div className="bg-muted rounded-lg p-2">
                                            <div className="text-xs text-muted-foreground">{t('training.loss') || 'Loss'}</div>
                                            <div className="text-lg font-semibold" style={{ color: selectedProcess.color }}>
                                                {selectedProcess.latestLoss?.toFixed(4) ?? '--'}
                                            </div>
                                        </div>
                                        <div className="bg-muted rounded-lg p-2">
                                            <div className="text-xs text-muted-foreground">{t('training.lr') || 'LR'}</div>
                                            <div className="text-lg font-semibold" style={{ color: selectedProcess.color }}>
                                                {selectedProcess.latestLR && selectedProcess.latestLR > 0
                                                    ? selectedProcess.latestLR.toExponential(2)
                                                    : t('training.lr-warmup') || 'warmup'
                                                }
                                            </div>
                                        </div>
                                    </div>
                                    {/* 进度信息卡片 */}
                                    <div className="bg-muted rounded-lg p-2 flex flex-col justify-center">
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-xs text-muted-foreground">{t('training.progress') || 'Progress'}</span>
                                            <span className="text-xs font-medium">{selectedProcess.progress ?? 0}%</span>
                                        </div>
                                        <div className="w-full bg-background rounded-full h-2">
                                            <div
                                                className="h-2 rounded-full transition-all duration-300"
                                                style={{
                                                    width: `${selectedProcess.progress ?? 0}%`,
                                                    backgroundColor: selectedProcess.color,
                                                }}
                                            />
                                        </div>
                                        <div className="flex items-center justify-between text-xs mt-2">
                                            <span className="text-muted-foreground">
                                                {t('training.elapsed-time') || 'Elapsed'}: {formatDuration(selectedProcess.latestElapsedTime)}
                                            </span>
                                            <span className="text-muted-foreground">
                                                {t('training.remaining-time') || 'Remaining'}: {formatDuration(selectedProcess.latestRemainingTime)}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* 图表控制栏 */}
                                <div className="px-3 pb-2 flex items-center justify-between shrink-0">
                                    <div className="flex items-center gap-4">
                                        <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={showRaw}
                                                onChange={(e) => setShowRaw(e.target.checked)}
                                                className="w-3.5 h-3.5 rounded border-border"
                                            />
                                            <span>{t('training.raw') || '原始'}</span>
                                        </label>
                                        <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={showSmoothed}
                                                onChange={(e) => setShowSmoothed(e.target.checked)}
                                                className="w-3.5 h-3.5 rounded border-border"
                                            />
                                            <span>{t('training.smoothed') || '平滑'}</span>
                                        </label>
                                    </div>
                                    <div className="text-xs text-muted-foreground">
                                        {selectedProcess.metrics.length} {t('training.data-points') || 'data points'}
                                    </div>
                                </div>

                                {/* 图表垂直排列区域 */}
                                <div className="flex-1 flex flex-col gap-3 px-3 pb-3 min-h-0 overflow-auto">
                                    {/* Loss 图表 */}
                                    <div 
                                        className="flex-1 min-h-[280px] flex flex-col bg-card rounded-xl border border-border shadow-sm overflow-hidden"
                                        onWheel={(e) => handleWheel(e, 'loss')}
                                        onMouseDown={(e) => handleMouseDown(e, 'loss')}
                                        onMouseMove={(e) => handleMouseMove(e, 'loss')}
                                        onMouseUp={() => handleMouseUp('loss')}
                                        onMouseLeave={() => handleMouseLeave('loss')}
                                        style={{ cursor: isDragging.loss ? 'grabbing' : 'grab' }}
                                    >
                                        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30 shrink-0">
                                            <div className="flex items-center gap-3">
                                                <span className="text-sm font-medium">
                                                    {t('training.loss-curve') || 'Loss Curve'}
                                                </span>
                                                <span className="text-xs text-muted-foreground">
                                                    Step {Math.round(zoomDomain.loss?.min ?? selectedProcess.metrics[0]?.step ?? 0)} - {Math.round(zoomDomain.loss?.max ?? selectedProcess.metrics[selectedProcess.metrics.length - 1]?.step ?? 0)}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <label className="flex items-center gap-1 text-xs cursor-pointer" title={t('training.log-scale') || 'Log Scale'}>
                                                    <input
                                                        type="checkbox"
                                                        checked={isLogScaleLoss}
                                                        onChange={(e) => setIsLogScaleLoss(e.target.checked)}
                                                        className="w-3 h-3 rounded border-border"
                                                    />
                                                    <span>log</span>
                                                </label>
                                                <button
                                                    onClick={() => handleFitDomain('loss')}
                                                    className="text-xs px-2 py-0.5 bg-background border border-border rounded hover:bg-muted transition-colors"
                                                    title={t('training.fit-domain') || 'Fit Domain'}
                                                >
                                                    {t('training.fit') || 'Fit'}
                                                </button>
                                                <button
                                                    onClick={() => toggleFullscreen('loss')}
                                                    className="text-xs px-2 py-0.5 bg-background border border-border rounded hover:bg-muted transition-colors flex items-center gap-1"
                                                    title={t('training.fullscreen') || 'Fullscreen'}
                                                >
                                                    <Maximize2 className="w-3 h-3" />
                                                </button>
                                            </div>
                                        </div>
                                        <div className="flex-1 min-h-0 p-2 relative" ref={chartRef}>
                                            <div className="absolute inset-2">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart 
                                                        data={selectedProcess.metrics}
                                                        margin={{ top: 10, right: 20, left: 50, bottom: 40 }}
                                                    >
                                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                                                        <XAxis
                                                            dataKey="step"
                                                            type="number"
                                                            domain={zoomDomain.loss ? [zoomDomain.loss.min, zoomDomain.loss.max] : ['dataMin', 'dataMax']}
                                                            tick={{ fontSize: 11, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => value.toLocaleString()}
                                                            stroke="var(--border)"
                                                            label={{ value: 'Step', position: 'insideBottom', offset: -25, fontSize: 11, fill: 'var(--muted-foreground)' }}
                                                            allowDataOverflow
                                                        />
                                                        <YAxis
                                                            type="number"
                                                            scale={isLogScaleLoss ? 'log' : 'linear'}
                                                            domain={calculateYDomain(
                                                                selectedProcess.metrics.map(m => ({ step: m.step, loss: m.loss })),
                                                                zoomDomain.loss,
                                                                'loss',
                                                                isLogScaleLoss
                                                            )}
                                                            tick={{ fontSize: 11, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => isLogScaleLoss ? value.toExponential(1) : value.toFixed(4)}
                                                            width={60}
                                                            stroke="var(--border)"
                                                            allowDataOverflow
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: 'var(--background)',
                                                                border: '1px solid var(--border)',
                                                                borderRadius: '6px',
                                                                fontSize: '12px',
                                                                color: 'var(--foreground)',
                                                            }}
                                                            formatter={(value: number, name: string) => [
                                                                isLogScaleLoss ? (value as number).toExponential(4) : (value as number).toFixed(4), 
                                                                name === 'raw' ? '原始' : '平滑'
                                                            ]}
                                                            labelFormatter={(label) => `Step: ${label}`}
                                                        />
                                                        {showRaw && (
                                                            <Line
                                                                type="monotone"
                                                                dataKey="loss"
                                                                name="raw"
                                                                stroke={selectedProcess.color}
                                                                strokeWidth={1.5}
                                                                dot={false}
                                                                isAnimationActive={false}
                                                                opacity={0.7}
                                                            />
                                                        )}
                                                        {showSmoothed && smoothingFactor > 0 && (
                                                            <Line
                                                                type="monotone"
                                                                dataKey={(data) => {
                                                                    const idx = selectedProcess.metrics.findIndex(m => m.step === data.step);
                                                                    return selectedProcess.smoothedMetrics[idx]?.loss ?? data.loss;
                                                                }}
                                                                name="smoothed"
                                                                stroke={selectedProcess.color}
                                                                strokeWidth={2.5}
                                                                dot={false}
                                                                isAnimationActive={false}
                                                            />
                                                        )}
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                        <div className="px-3 py-1 text-xs text-muted-foreground text-center border-t border-border bg-muted/20 shrink-0">
                                            {t('training.drag-hint') || 'Drag to pan, Alt+Scroll to zoom'}
                                        </div>
                                    </div>

                                    {/* LR 图表 */}
                                    <div 
                                        className="flex-1 min-h-[280px] flex flex-col bg-card rounded-xl border border-border shadow-sm overflow-hidden"
                                        onWheel={(e) => handleWheel(e, 'lr')}
                                        onMouseDown={(e) => handleMouseDown(e, 'lr')}
                                        onMouseMove={(e) => handleMouseMove(e, 'lr')}
                                        onMouseUp={() => handleMouseUp('lr')}
                                        onMouseLeave={() => handleMouseLeave('lr')}
                                        style={{ cursor: isDragging.lr ? 'grabbing' : 'grab' }}
                                    >
                                        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30 shrink-0">
                                            <div className="flex items-center gap-3">
                                                <span className="text-sm font-medium">
                                                    {t('training.lr-curve') || 'Learning Rate Curve'}
                                                </span>
                                                <span className="text-xs text-muted-foreground">
                                                    Step {Math.round(zoomDomain.lr?.min ?? selectedProcess.metrics[0]?.step ?? 0)} - {Math.round(zoomDomain.lr?.max ?? selectedProcess.metrics[selectedProcess.metrics.length - 1]?.step ?? 0)}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <label className="flex items-center gap-1 text-xs cursor-pointer" title={t('training.log-scale') || 'Log Scale'}>
                                                    <input
                                                        type="checkbox"
                                                        checked={isLogScaleLR}
                                                        onChange={(e) => setIsLogScaleLR(e.target.checked)}
                                                        className="w-3 h-3 rounded border-border"
                                                    />
                                                    <span>log</span>
                                                </label>
                                                <button
                                                    onClick={() => handleFitDomain('lr')}
                                                    className="text-xs px-2 py-0.5 bg-background border border-border rounded hover:bg-muted transition-colors"
                                                    title={t('training.fit-domain') || 'Fit Domain'}
                                                >
                                                    {t('training.fit') || 'Fit'}
                                                </button>
                                                <button
                                                    onClick={() => toggleFullscreen('lr')}
                                                    className="text-xs px-2 py-0.5 bg-background border border-border rounded hover:bg-muted transition-colors flex items-center gap-1"
                                                    title={t('training.fullscreen') || 'Fullscreen'}
                                                >
                                                    <Maximize2 className="w-3 h-3" />
                                                </button>
                                            </div>
                                        </div>
                                        <div className="flex-1 min-h-0 p-2 relative">
                                            <div className="absolute inset-2">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart 
                                                        data={selectedProcess.metrics}
                                                        margin={{ top: 10, right: 20, left: 50, bottom: 40 }}
                                                    >
                                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                                                        <XAxis
                                                            dataKey="step"
                                                            type="number"
                                                            domain={zoomDomain.lr ? [zoomDomain.lr.min, zoomDomain.lr.max] : ['dataMin', 'dataMax']}
                                                            tick={{ fontSize: 11, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => value.toLocaleString()}
                                                            stroke="var(--border)"
                                                            label={{ value: 'Step', position: 'insideBottom', offset: -25, fontSize: 11, fill: 'var(--muted-foreground)' }}
                                                            allowDataOverflow
                                                        />
                                                        <YAxis
                                                            type="number"
                                                            scale={isLogScaleLR ? 'log' : 'linear'}
                                                            domain={calculateYDomain(
                                                                selectedProcess.metrics.map(m => ({ step: m.step, lr: m.lr })),
                                                                zoomDomain.lr,
                                                                'lr',
                                                                isLogScaleLR
                                                            )}
                                                            tick={{ fontSize: 11, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => value === 0 ? '0' : (isLogScaleLR ? value.toExponential(1) : value.toExponential(4))}
                                                            width={60}
                                                            stroke="var(--border)"
                                                            allowDataOverflow
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: 'var(--background)',
                                                                border: '1px solid var(--border)',
                                                                borderRadius: '6px',
                                                                fontSize: '12px',
                                                                color: 'var(--foreground)',
                                                            }}
                                                            formatter={(value: number) => [
                                                                value === 0 ? '0 (warmup)' : (isLogScaleLR ? value.toExponential(4) : value.toExponential(4)), 
                                                                'LR'
                                                            ]}
                                                            labelFormatter={(label) => `Step: ${label}`}
                                                        />
                                                        <Line
                                                            type="monotone"
                                                            dataKey="lr"
                                                            name="lr"
                                                            stroke={selectedProcess.color}
                                                            strokeWidth={2}
                                                            dot={false}
                                                            isAnimationActive={false}
                                                        />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                        <div className="px-3 py-1 text-xs text-muted-foreground text-center border-t border-border bg-muted/20 shrink-0">
                                            {t('training.drag-hint') || 'Drag to pan, Alt+Scroll to zoom'}
                                        </div>
                                    </div>
                                </div>

                                {/* 全屏 Dialog */}
                                <Dialog open={!!fullscreenChart} onOpenChange={() => setFullscreenChart(null)}>
                                    <DialogContent className="max-w-[90vw] w-[1200px] h-[85vh] p-0 flex flex-col">
                                        <DialogHeader className="px-4 py-3 border-b shrink-0">
                                            <DialogTitle className="flex items-center justify-between">
                                                <span>
                                                    {fullscreenChart === 'loss' 
                                                        ? (t('training.loss-curve') || 'Loss Curve')
                                                        : (t('training.lr-curve') || 'Learning Rate Curve')
                                                    }
                                                </span>
                                                <button
                                                    onClick={() => setFullscreenChart(null)}
                                                    className="text-xs px-3 py-1.5 bg-background border border-border rounded hover:bg-muted transition-colors flex items-center gap-1"
                                                >
                                                    <Minimize2 className="w-3 h-3" />
                                                    {t('training.exit-fullscreen') || 'Exit'}
                                                </button>
                                            </DialogTitle>
                                        </DialogHeader>
                                        <div className="flex-1 p-4 min-h-0">
                                            {fullscreenChart === 'loss' && selectedProcess && (
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart 
                                                        data={selectedProcess.metrics}
                                                        margin={{ top: 20, right: 30, left: 60, bottom: 50 }}
                                                    >
                                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                                                        <XAxis
                                                            dataKey="step"
                                                            type="number"
                                                            domain={zoomDomain.loss ? [zoomDomain.loss.min, zoomDomain.loss.max] : ['dataMin', 'dataMax']}
                                                            tick={{ fontSize: 12, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => value.toLocaleString()}
                                                            stroke="var(--border)"
                                                            label={{ value: 'Step', position: 'insideBottom', offset: -35, fontSize: 12, fill: 'var(--muted-foreground)' }}
                                                            allowDataOverflow
                                                        />
                                                        <YAxis
                                                            type="number"
                                                            scale={isLogScaleLoss ? 'log' : 'linear'}
                                                            domain={calculateYDomain(
                                                                selectedProcess.metrics.map(m => ({ step: m.step, loss: m.loss })),
                                                                zoomDomain.loss,
                                                                'loss',
                                                                isLogScaleLoss
                                                            )}
                                                            tick={{ fontSize: 12, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => isLogScaleLoss ? value.toExponential(1) : value.toFixed(4)}
                                                            width={70}
                                                            stroke="var(--border)"
                                                            allowDataOverflow
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: 'var(--background)',
                                                                border: '1px solid var(--border)',
                                                                borderRadius: '6px',
                                                                fontSize: '13px',
                                                                color: 'var(--foreground)',
                                                            }}
                                                            formatter={(value: number, name: string) => [
                                                                isLogScaleLoss ? (value as number).toExponential(4) : (value as number).toFixed(4), 
                                                                name === 'raw' ? '原始' : '平滑'
                                                            ]}
                                                            labelFormatter={(label) => `Step: ${label}`}
                                                        />
                                                        {showRaw && (
                                                            <Line
                                                                type="monotone"
                                                                dataKey="loss"
                                                                name="raw"
                                                                stroke={selectedProcess.color}
                                                                strokeWidth={2}
                                                                dot={false}
                                                                isAnimationActive={false}
                                                                opacity={0.7}
                                                            />
                                                        )}
                                                        {showSmoothed && smoothingFactor > 0 && (
                                                            <Line
                                                                type="monotone"
                                                                dataKey={(data) => {
                                                                    const idx = selectedProcess.metrics.findIndex(m => m.step === data.step);
                                                                    return selectedProcess.smoothedMetrics[idx]?.loss ?? data.loss;
                                                                }}
                                                                name="smoothed"
                                                                stroke={selectedProcess.color}
                                                                strokeWidth={3}
                                                                dot={false}
                                                                isAnimationActive={false}
                                                            />
                                                        )}
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            )}
                                            {fullscreenChart === 'lr' && selectedProcess && (
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart 
                                                        data={selectedProcess.metrics}
                                                        margin={{ top: 20, right: 30, left: 60, bottom: 50 }}
                                                    >
                                                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                                                        <XAxis
                                                            dataKey="step"
                                                            type="number"
                                                            domain={zoomDomain.lr ? [zoomDomain.lr.min, zoomDomain.lr.max] : ['dataMin', 'dataMax']}
                                                            tick={{ fontSize: 12, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => value.toLocaleString()}
                                                            stroke="var(--border)"
                                                            label={{ value: 'Step', position: 'insideBottom', offset: -35, fontSize: 12, fill: 'var(--muted-foreground)' }}
                                                            allowDataOverflow
                                                        />
                                                        <YAxis
                                                            type="number"
                                                            scale={isLogScaleLR ? 'log' : 'linear'}
                                                            domain={calculateYDomain(
                                                                selectedProcess.metrics.map(m => ({ step: m.step, lr: m.lr })),
                                                                zoomDomain.lr,
                                                                'lr',
                                                                isLogScaleLR
                                                            )}
                                                            tick={{ fontSize: 12, fill: 'var(--foreground)' }}
                                                            tickFormatter={(value) => value === 0 ? '0' : (isLogScaleLR ? value.toExponential(1) : value.toExponential(4))}
                                                            width={70}
                                                            stroke="var(--border)"
                                                            allowDataOverflow
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: 'var(--background)',
                                                                border: '1px solid var(--border)',
                                                                borderRadius: '6px',
                                                                fontSize: '13px',
                                                                color: 'var(--foreground)',
                                                            }}
                                                            formatter={(value: number) => [
                                                                value === 0 ? '0 (warmup)' : (isLogScaleLR ? value.toExponential(4) : value.toExponential(4)), 
                                                                'LR'
                                                            ]}
                                                            labelFormatter={(label) => `Step: ${label}`}
                                                        />
                                                        <Line
                                                            type="monotone"
                                                            dataKey="lr"
                                                            name="lr"
                                                            stroke={selectedProcess.color}
                                                            strokeWidth={2.5}
                                                            dot={false}
                                                            isAnimationActive={false}
                                                        />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            )}
                                        </div>
                                    </DialogContent>
                                </Dialog>

                                {/* Loss 异常警告 Modal */}
                                <Dialog 
                                    open={!!activeWarning} 
                                    onOpenChange={(open) => {
                                        if (!open && activeWarning) {
                                            setDismissedWarnings(prev => new Set([...prev, activeWarning.runId]));
                                            setActiveWarning(null);
                                        }
                                    }}
                                >
                                    <DialogContent className="max-w-md border-red-200">
                                        <DialogHeader>
                                            <DialogTitle className="flex items-center gap-2 text-red-600">
                                                <AlertTriangle className="w-6 h-6" />
                                                {activeWarning?.type === 'nan' 
                                                    ? (t('training.loss-warning-nan-title') || 'Loss 异常：NaN') 
                                                    : (t('training.loss-warning-high-title') || 'Loss 异常：数值过高')
                                                }
                                            </DialogTitle>
                                        </DialogHeader>
                                        
                                        <div className="space-y-4 py-4">
                                            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                                <p className="text-sm text-red-800 font-medium">
                                                    {t('training.loss-warning-current') || '当前 Loss'}: {activeWarning?.type === 'nan' ? 'NaN' : activeWarning?.value.toFixed(2)}
                                                </p>
                                                <p className="text-sm text-red-600 mt-1">
                                                    {t('training.loss-warning-message') || '检测到异常 loss 值，训练可能出现问题。'}
                                                </p>
                                            </div>
                                            
                                            <div>
                                                <h4 className="text-sm font-semibold mb-2">{t('training.loss-warning-suggestions') || '建议采取的措施：'}</h4>
                                                <ul className="space-y-2">
                                                    {activeWarning?.suggestions.map((suggestion, index) => (
                                                        <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                                                            <span className="text-primary font-medium">{index + 1}.</span>
                                                            <span>{suggestion}</span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>
                                        
                                        <div className="flex justify-end">
                                            <Button 
                                                onClick={() => {
                                                    if (activeWarning) {
                                                        setDismissedWarnings(prev => new Set([...prev, activeWarning.runId]));
                                                        setActiveWarning(null);
                                                    }
                                                }}
                                                variant="default"
                                            >
                                                {t('training.loss-warning-dismiss') || '我知道了'}
                                            </Button>
                                        </div>
                                    </DialogContent>
                                </Dialog>

                                {/* 询问AI对话框 */}
                                <Dialog 
                                    open={isAskAIDialogOpen} 
                                    onOpenChange={(open) => {
                                        setIsAskAIDialogOpen(open);
                                        if (!open) {
                                            setCustomQuestion('');
                                        }
                                    }}
                                >
                                    <DialogContent className="max-w-lg">
                                        <DialogHeader>
                                            <DialogTitle className="flex items-center gap-2">
                                                <Sparkles className="w-5 h-5 text-primary" />
                                                {t('ask-ai-title') || '询问AI分析训练曲线'}
                                            </DialogTitle>
                                        </DialogHeader>
                                        
                                        <div className="space-y-4 py-4">
                                            <div className="space-y-2">
                                                <Label htmlFor="custom-question">
                                                    {t('ask-ai-question-label') || '您想询问什么问题？（可选）'}
                                                </Label>
                                                <Textarea
                                                    id="custom-question"
                                                    placeholder={t('ask-ai-placeholder') || '例如：这个曲线的趋势正常吗？是否存在过拟合？'}
                                                    value={customQuestion}
                                                    onChange={(e) => setCustomQuestion(e.target.value)}
                                                    rows={3}
                                                    className="resize-none"
                                                />
                                                <p className="text-xs text-muted-foreground">
                                                    {t('ask-ai-hint') || '留空将使用默认问题：请帮我分析这个训练曲线的趋势和是否存在异常。'}
                                                </p>
                                            </div>
                                            
                                            {selectedProcess && (
                                                <div className="bg-muted rounded-lg p-3 space-y-1">
                                                    <p className="text-xs font-medium text-muted-foreground">
                                                        {t('current-info') || '当前训练信息'}
                                                    </p>
                                                    <div className="text-xs text-muted-foreground space-y-1">
                                                        <p>PID: {selectedProcess.pid}</p>
                                                        <p>{t('training.step') || '步数'}: {selectedProcess.latestStep}</p>
                                                        <p>{t('training.loss') || 'Loss'}: {selectedProcess.latestLoss?.toFixed(6) ?? '--'}</p>
                                                        <p>{t('training.lr') || 'LR'}: {selectedProcess.latestLR && selectedProcess.latestLR > 0 ? selectedProcess.latestLR.toExponential(2) : 'warmup'}</p>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        
                                        <div className="flex justify-end gap-2">
                                            <Button 
                                                variant="outline"
                                                onClick={() => {
                                                    setIsAskAIDialogOpen(false);
                                                    setCustomQuestion('');
                                                }}
                                                disabled={isAskingAI}
                                            >
                                                {t('action.cancel') || '取消'}
                                            </Button>
                                            <Button 
                                                onClick={handleAskAI}
                                                disabled={isAskingAI || !selectedProcess}
                                                className="gap-2"
                                            >
                                                {isAskingAI ? (
                                                    <>
                                                        <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                                                        {t('asking-ai') || '发送中...'}
                                                    </>
                                                ) : (
                                                    <>
                                                        <Sparkles className="w-4 h-4" />
                                                        {t('send-to-ai') || '发送给AI'}
                                                    </>
                                                )}
                                            </Button>
                                        </div>
                                    </DialogContent>
                                </Dialog>
                            </>
                        ) : (
                            <div className="flex-1 flex items-center justify-center p-4">
                                {processDataMap.size === 0 ? (
                                    <div className="max-w-[420px] rounded-xl border border-border bg-card px-6 py-7 text-center shadow-sm">
                                        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-500/16 dark:text-indigo-200">
                                            <Activity className="h-6 w-6" />
                                        </div>
                                        <div className="text-sm font-semibold text-foreground">
                                            {isWaitingForMetricWrite
                                                ? t('training.empty-waiting-title', { defaultValue: '训练已启动，等待指标写入' })
                                                : t('training.empty-title', { defaultValue: '还没有训练指标' })}
                                        </div>
                                        <p className="mt-2 text-xs leading-5 text-muted-foreground">
                                            {isWaitingForMetricWrite
                                                ? t('training.empty-waiting-desc', { defaultValue: '已检测到训练进程，loss、lr 等指标写入后会自动刷新到这里。' })
                                                : t('training.empty-desc', { defaultValue: '请在对话中使用“监控训练”，系统获取到训练进程、loss、lr 等信息后，会在这里展示曲线和指标。' })}
                                        </p>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={onMonitorTraining}
                                            disabled={!onMonitorTraining}
                                            className="mt-4 inline-flex h-8 items-center gap-2 rounded-full border-indigo-200/70 bg-indigo-50 px-3 py-1.5 text-xs font-medium text-indigo-700 hover:bg-indigo-100 dark:border-indigo-400/20 dark:bg-indigo-500/10 dark:text-indigo-200 dark:hover:bg-indigo-500/20"
                                        >
                                            <MessageSquareText className="h-3.5 w-3.5" />
                                            {t('training.empty-command', { defaultValue: '监控训练' })}
                                        </Button>
                                    </div>
                                ) : (
                                    <div className="text-center text-muted-foreground text-sm">
                                        {t('training.select-process') || 'Select a process to view'}
                                    </div>
                                )}
                            </div>
                        )
                    )}
                </div>
            </div>
        </div>
    );
};

export default memo(TrainingMetricsPanel);
