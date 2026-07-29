import { Button, Modal, Spin } from 'antd';
import { ArrowRight, BookOpen, Compass, FolderKanban, MessagesSquare, PlayCircle, Rocket } from 'lucide-react';
import dayjs from 'dayjs';
import { useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import {
    Bar,
    BarChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { trpc } from '@/api/trpc';
import extended from '@/pages/ContentPage/utils.ts';
import { useOverviewRoom } from '@/context/OverviewRoomContext.tsx';
import { useFirstTimeGuide } from '@/context/FirstTimeGuideContext.tsx';
import { useIsMobile } from '@/hooks/use-mobile';
import { useTheme } from '@/context/ThemeContext.tsx';

const ONBOARDING_CENTER_KEY = 'medflow_onboarding_center_v1';

type OnboardingCenterState = {
    onboardingCenterSeen: boolean;
    pageTourCompleted: boolean;
    textGuideCompleted: boolean;
    practiceCompleted: boolean;
    neverAskAgain: boolean;
};

const getOnboardingCenterState = (): OnboardingCenterState => {
    try {
        const stored = localStorage.getItem(ONBOARDING_CENTER_KEY);
        if (!stored) {
            return {
                onboardingCenterSeen: false,
                pageTourCompleted: false,
                textGuideCompleted: false,
                practiceCompleted: false,
                neverAskAgain: false,
            };
        }

        return {
            onboardingCenterSeen: false,
            pageTourCompleted: false,
            textGuideCompleted: false,
            practiceCompleted: false,
            neverAskAgain: false,
            ...JSON.parse(stored),
        };
    } catch {
        return { onboardingCenterSeen: false, neverAskAgain: false };
    }
};

const saveOnboardingCenterState = (state: Partial<OnboardingCenterState>) => {
    const next = {
        ...getOnboardingCenterState(),
        ...state,
    };
    localStorage.setItem(ONBOARDING_CENTER_KEY, JSON.stringify(next));
};

type StatCardProps = {
    title: string;
    value: string | number;
    hint: string;
    icon: ReactNode;
    featured?: boolean;
    compact?: boolean;
};

const StatCard = ({ title, value, hint, icon, featured = false, compact = false }: StatCardProps) => {
    return (
        <div
            className={`relative overflow-hidden rounded-[18px] transition-all duration-200 hover:-translate-y-0.5 ${
                compact ? 'p-2' : 'p-3'
            } ${
                featured
                    ? 'border border-sky-100/90 bg-[linear-gradient(180deg,rgba(247,251,255,0.98)_0%,rgba(255,255,255,0.95)_100%)] shadow-[0_22px_44px_-34px_rgba(14,165,233,0.2)] hover:shadow-[0_26px_50px_-34px_rgba(14,165,233,0.24)] dark:border-sky-400/20 dark:bg-[linear-gradient(180deg,rgba(14,24,39,0.94)_0%,rgba(15,23,42,0.92)_100%)] dark:shadow-[0_22px_46px_-34px_rgba(8,47,73,0.58)] dark:hover:shadow-[0_26px_54px_-34px_rgba(14,165,233,0.22)]'
                    : 'border border-white/85 bg-[linear-gradient(180deg,rgba(255,255,255,0.9)_0%,rgba(239,246,255,0.92)_100%)] shadow-[0_22px_46px_-34px_rgba(15,23,42,0.26)] hover:shadow-[0_26px_54px_-34px_rgba(15,23,42,0.3)] dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(17,24,39,0.94)_0%,rgba(15,23,42,0.92)_100%)] dark:shadow-[0_22px_46px_-34px_rgba(2,6,23,0.7)] dark:hover:shadow-[0_26px_54px_-34px_rgba(15,23,42,0.78)]'
            }`}
        >
            <div
                className={`pointer-events-none absolute -right-10 -top-12 h-24 w-24 rounded-full blur-3xl ${
                    featured ? 'bg-sky-200/20 dark:bg-sky-400/14' : 'bg-sky-200/25 dark:bg-sky-500/10'
                }`}
            />
            <div className={`flex items-center justify-between ${compact ? 'mb-1' : 'mb-1.5'}`}>
                <div className={`${compact ? 'text-[10px]' : 'text-[12px]'} font-semibold uppercase tracking-[0.05em] ${featured ? 'text-sky-600 dark:text-sky-300' : 'text-slate-500 dark:text-slate-400'}`}>{title}</div>
                <div
                    className={`flex items-center justify-center shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] ${
                        compact ? 'h-7 w-7 rounded-[10px]' : 'h-8 w-8 rounded-[11px]'
                    } ${
                        featured ? 'bg-sky-100 text-sky-600 dark:bg-sky-500/14 dark:text-sky-300 dark:shadow-none' : 'bg-sky-100 text-sky-700 dark:bg-slate-800 dark:text-sky-300 dark:shadow-none'
                    }`}
                >
                    {icon}
                </div>
            </div>
            <div className={`${compact ? 'text-[20px]' : 'text-[26px]'} font-semibold tracking-tight ${featured ? 'text-slate-950 dark:text-slate-50' : 'text-slate-900 dark:text-slate-100'}`}>{value}</div>
            <div className={`mt-0.5 ${compact ? 'text-[10px] leading-4' : 'text-[12px] leading-5'} ${featured ? 'text-slate-600 dark:text-slate-300' : 'text-slate-500 dark:text-slate-400'}`}>{hint}</div>
        </div>
    );
};

type ProjectStatusDotProps = {
    active: boolean;
};

const ProjectStatusDot = ({ active }: ProjectStatusDotProps) => (
    <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${
            active ? 'bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.12)]' : 'bg-slate-300'
        }`}
    />
);

interface MonthlyRunItem {
    month: string;
    count: number;
}

const LandingPage = () => {
    const navigate = useNavigate();
    const { overviewData } = useOverviewRoom();
    const { startTour } = useFirstTimeGuide();
    const { t } = useTranslation();
    const isMobile = useIsMobile();
    const { resolvedTheme } = useTheme();
    const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);

    const { data: projectsResponse, isLoading: isProjectsLoading } =
        trpc.getProjects.useQuery({
            pagination: {
                page: 1,
                pageSize: 200,
            },
            sort: {
                field: 'createdAt',
                order: 'desc',
            },
        });

    const { data: latestRunResponse, isLoading: isLatestRunLoading } =
        trpc.getLatestRun.useQuery();

    const {
        data: latestRunnableRunResponse,
        isLoading: isLatestRunnableRunLoading,
    } = trpc.getLatestRunnableRun.useQuery();

    const projects = projectsResponse?.data?.list ?? [];
    const latestRun = latestRunResponse?.data ?? null;
    const latestRunnableRun = latestRunnableRunResponse?.data ?? null;
    const visibleProjects = isMobile ? projects.slice(0, 3) : projects.slice(0, 4);

    const summary = useMemo(() => {
        return projects.reduce(
            (acc, project) => {
                acc.projects += 1;
                acc.runs += project.total;
                acc.running += project.running;
                acc.pending += project.pending || 0;
                acc.finished += project.finished;
                return acc;
            },
            {
                projects: 0,
                runs: 0,
                running: 0,
                pending: 0,
                finished: 0,
            },
        );
    }, [projects]);

    const isLoading = isProjectsLoading || isLatestRunLoading || isLatestRunnableRunLoading;
    const monthlyRuns: MonthlyRunItem[] = overviewData?.monthlyRuns
        ? JSON.parse(overviewData.monthlyRuns)
        : [];
    const chartData = [...monthlyRuns].reverse().slice(-6);
    const yAxisMax = chartData.length
        ? Math.max(...chartData.map((item) => item.count))
        : 0;
    const yAxisMin = chartData.length
        ? Math.min(...chartData.map((item) => item.count))
        : 0;
    const ticks = chartData.length ? extended(yAxisMin, yAxisMax, 4) : [0];
    const safeTicks = ticks.length ? ticks : [0];
    const maxTick = safeTicks[safeTicks.length - 1] ?? 0;
    const yAxisWidth =
        maxTick < 10 ? 20 : maxTick < 100 ? 25 : maxTick < 1000 ? 30 : 42;
    const formatMonthLabel = (value: string) => {
        const numericMonth = parseInt(value.split('-')[1], 10);
        if (Number.isNaN(numericMonth) || numericMonth < 1 || numericMonth > 12) {
            return value;
        }
        return t(`home.landing.chart.month-${numericMonth}`);
    };
    const formatFriendlyDateTime = (value?: string | null) => {
        if (!value) return '--';
        const parsed = dayjs(value);
        return parsed.isValid() ? parsed.format('YYYY-MM-DD HH:mm') : value;
    };
    const hasOverviewData = projects.length > 0 || chartData.length > 0 || !!latestRun;
    const getStartedDisabled = !latestRunnableRun || isLoading;

    const enterLatestRunnableRun = () => {
        if (!latestRunnableRun) return;
        navigate(`/projects/${latestRunnableRun.project}/runs/${latestRunnableRun.runId}`);
    };

    const handleEnterRunPage = () => {
        if (!latestRunnableRun) return;

        const onboardingState = getOnboardingCenterState();
        if (onboardingState.neverAskAgain) {
            enterLatestRunnableRun();
            return;
        }

        setIsOnboardingOpen(true);
    };

    const handleStartPageTour = () => {
        if (!latestRunnableRun) return;
        saveOnboardingCenterState({
            onboardingCenterSeen: true,
            pageTourCompleted: true,
        });
        setIsOnboardingOpen(false);
        startTour(latestRunnableRun.project);
    };

    const handleStartTrainingGuide = () => {
        if (!latestRunnableRun) return;
        saveOnboardingCenterState({
            onboardingCenterSeen: true,
            textGuideCompleted: true,
        });
        sessionStorage.setItem('medflow_open_training_guide', 'true');
        setIsOnboardingOpen(false);
        enterLatestRunnableRun();
    };

    const handleStartPractice = () => {
        if (!latestRunnableRun) return;
        saveOnboardingCenterState({
            onboardingCenterSeen: true,
            practiceCompleted: true,
        });
        sessionStorage.setItem('medflow_open_quick_start', 'true');
        setIsOnboardingOpen(false);
        enterLatestRunnableRun();
    };

    const handleContinueWithoutGuide = () => {
        saveOnboardingCenterState({
            onboardingCenterSeen: true,
            neverAskAgain: true,
        });
        setIsOnboardingOpen(false);
        enterLatestRunnableRun();
    };

    const pageClassName = isMobile
        ? 'h-screen w-full overflow-x-hidden overflow-y-auto bg-[radial-gradient(circle_at_12%_8%,_rgba(37,99,235,0.14),_transparent_26%),radial-gradient(circle_at_88%_12%,_rgba(14,165,233,0.1),_transparent_24%),linear-gradient(180deg,_#f1f7ff_0%,_#f8fbff_44%,_#f3f7fd_100%)] dark:bg-[radial-gradient(circle_at_12%_8%,_rgba(37,99,235,0.18),_transparent_24%),radial-gradient(circle_at_88%_12%,_rgba(14,165,233,0.12),_transparent_24%),linear-gradient(180deg,_#020617_0%,_#0f172a_48%,_#111827_100%)]'
        : 'h-full w-full overflow-auto bg-[radial-gradient(circle_at_12%_8%,_rgba(37,99,235,0.14),_transparent_26%),radial-gradient(circle_at_88%_12%,_rgba(14,165,233,0.1),_transparent_24%),linear-gradient(180deg,_#f1f7ff_0%,_#f8fbff_44%,_#f3f7fd_100%)] dark:bg-[radial-gradient(circle_at_12%_8%,_rgba(37,99,235,0.18),_transparent_24%),radial-gradient(circle_at_88%_12%,_rgba(14,165,233,0.12),_transparent_24%),linear-gradient(180deg,_#020617_0%,_#0f172a_48%,_#111827_100%)]';
    const shellClassName = isMobile
        ? 'flex h-full min-h-full w-full flex-col px-2 py-2'
        : 'mx-auto flex min-h-full w-full max-w-[960px] flex-col px-4 py-4 md:px-5 lg:py-4 xl:max-w-[1040px] xl:px-6 xl:py-6 2xl:max-w-[1120px] 2xl:justify-center 2xl:px-8 2xl:py-10';
    const heroCardClassName = isMobile
        ? 'mt-2 w-full rounded-[20px] border border-white/85 bg-[linear-gradient(180deg,rgba(255,255,255,0.92)_0%,rgba(239,246,255,0.9)_100%)] p-2.5 shadow-[0_26px_72px_-40px_rgba(15,23,42,0.34)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.94)_0%,rgba(17,24,39,0.92)_100%)] dark:shadow-[0_26px_72px_-40px_rgba(2,6,23,0.82)]'
        : 'mt-2 rounded-[24px] border border-white/85 bg-[linear-gradient(180deg,rgba(255,255,255,0.92)_0%,rgba(239,246,255,0.9)_100%)] p-3 shadow-[0_26px_72px_-40px_rgba(15,23,42,0.34)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.94)_0%,rgba(17,24,39,0.92)_100%)] dark:shadow-[0_26px_72px_-40px_rgba(2,6,23,0.82)] md:p-3 lg:p-2.5 xl:mt-3 xl:p-4 2xl:mt-6 2xl:p-5';
    const contentGapClassName = isMobile ? 'mt-1.5' : 'mt-1 xl:mt-2 2xl:mt-3';
    const chartGridStroke = resolvedTheme === 'dark'
        ? 'rgba(148,163,184,0.18)'
        : 'rgba(125,147,178,0.22)';
    const chartCursorFill = resolvedTheme === 'dark'
        ? 'rgba(148, 163, 184, 0.12)'
        : 'rgba(148, 163, 184, 0.08)';
    const tooltipContentStyle = resolvedTheme === 'dark'
        ? {
              borderRadius: 14,
              border: '1px solid rgba(71,85,105,0.85)',
              background: 'linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(17,24,39,0.98) 100%)',
              boxShadow: '0 18px 36px -28px rgba(2,6,23,0.9)',
              color: '#e2e8f0',
          }
        : {
              borderRadius: 14,
              border: '1px solid rgba(203,213,225,0.9)',
              background: 'linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,250,252,0.96) 100%)',
              boxShadow: '0 18px 36px -28px rgba(15,23,42,0.28)',
          };

    return (
        <div className={pageClassName}>
            <div className={shellClassName}>
                <div className={heroCardClassName}>
                    <div className="flex flex-col gap-1.5 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-3xl">
                            <h1 className={`max-w-2xl font-semibold tracking-tight text-slate-950 dark:text-slate-50 ${isMobile ? 'text-[20px]' : 'text-[22px] lg:text-[28px] xl:text-[34px] 2xl:text-[38px]'}`}>
                                {t('home.landing.title')}
                            </h1>
                            <p className={`mt-0 max-w-2xl text-slate-500 dark:text-slate-400 ${isMobile ? 'text-[11px] leading-4' : 'text-[12px] leading-5 lg:text-[11px] lg:leading-4 xl:text-[12px] xl:leading-5'}`}>
                                {t('home.landing.subtitle')}
                            </p>
                        </div>

                        <div className="flex justify-end">
                            <Button
                                type="primary"
                                size="large"
                                onClick={handleEnterRunPage}
                                disabled={getStartedDisabled}
                                loading={isLatestRunnableRunLoading}
                                icon={<ArrowRight className="h-4 w-4" />}
                                className={`h-9 rounded-[14px] border-0 px-4.5 text-[14px] font-medium ${
                                    getStartedDisabled
                                        ? 'bg-slate-200 text-slate-400 shadow-none hover:scale-100 dark:bg-slate-800 dark:text-slate-500'
                                        : 'bg-[linear-gradient(180deg,#1f8be5_0%,#1778cf_100%)] shadow-[0_18px_32px_-24px_rgba(37,99,235,0.58)] hover:scale-[1.01] hover:shadow-[0_22px_36px_-24px_rgba(37,99,235,0.62)]'
                                }`}
                                title={getStartedDisabled ? t('home.landing.getStarted.disabled') : undefined}
                            >
                                {t('home.landing.enter')}
                            </Button>
                        </div>
                    </div>

                    <div className="mt-2 grid grid-cols-3 gap-2">
                        <StatCard
                            title={t('home.landing.cards.projects.title')}
                            value={summary.projects}
                            hint={t('home.landing.cards.projects.hint')}
                            icon={<FolderKanban className="h-5 w-5" />}
                            featured
                            compact={isMobile}
                        />
                        <StatCard
                            title={t('home.landing.cards.runs.title')}
                            value={summary.running + summary.pending}
                            hint={t('home.landing.cards.runs.hint', {
                                running: summary.running,
                                pending: summary.pending,
                            })}
                            icon={<PlayCircle className="h-5 w-5" />}
                            compact={isMobile}
                        />
                        <StatCard
                            title={t('home.landing.cards.finished.title')}
                            value={summary.runs}
                            hint={t('home.landing.cards.finished.hint', {
                                finished: summary.finished,
                                active: summary.running + summary.pending,
                            })}
                            icon={<MessagesSquare className="h-5 w-5" />}
                            compact={isMobile}
                        />
                    </div>
                </div>

                {!hasOverviewData ? (
                    <div className={`${contentGapClassName} rounded-[22px] border border-dashed border-border bg-card px-5 py-7 text-center shadow-sm dark:border-white/10 dark:bg-slate-900/70`}>
                        <div className="mx-auto max-w-xl">
                            <div className="text-lg font-semibold text-foreground dark:text-slate-100">
                                {t('home.landing.empty.title')}
                            </div>
                            <div className="mt-2 text-[12px] leading-6 text-muted-foreground dark:text-slate-400">
                                {t('home.landing.empty.description')}
                            </div>
                        </div>
                    </div>
                ) : (
                <div className={`${contentGapClassName} grid gap-2 lg:grid-cols-[1.12fr_0.88fr] xl:gap-3 2xl:gap-4`}>
                    <div className="rounded-[22px] border border-white/85 bg-[linear-gradient(180deg,rgba(255,255,255,0.94)_0%,rgba(248,250,252,0.96)_100%)] p-3 shadow-[0_20px_52px_-36px_rgba(15,23,42,0.32)] dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.95)_0%,rgba(17,24,39,0.96)_100%)] dark:shadow-[0_20px_52px_-36px_rgba(2,6,23,0.75)] lg:p-2.5 xl:p-3.5 2xl:p-4">
                        <div className="mb-1.5 flex items-center justify-between">
                            <div>
                                <div className="text-[15px] font-semibold text-foreground dark:text-slate-100">
                                    {t('home.landing.projects.title')}
                                </div>
                            </div>
                            {isLoading ? <Spin size="small" /> : null}
                        </div>

                        <div className={`${isMobile ? 'space-y-1' : 'space-y-1 lg:space-y-0.5 xl:space-y-1.5'}`}>
                            {visibleProjects.map((project, index) => (
                                <div
                                    key={project.project}
                                    className={`flex w-full items-center justify-between rounded-[16px] text-left transition-all duration-200 ${isMobile ? 'px-3 py-1.5' : 'px-3.5 py-1.5'} ${
                                        index === 0
                                            ? 'border border-sky-200/90 bg-[linear-gradient(180deg,rgba(239,246,255,0.98)_0%,rgba(241,245,249,0.96)_100%)] shadow-[0_18px_36px_-30px_rgba(14,165,233,0.28)] dark:border-sky-400/25 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96)_0%,rgba(30,41,59,0.9)_100%)] dark:shadow-[0_18px_36px_-30px_rgba(8,47,73,0.64)]'
                                            : 'border border-slate-200/75 bg-[linear-gradient(180deg,rgba(248,250,252,0.92)_0%,rgba(241,245,249,0.92)_100%)] shadow-[0_12px_28px_-26px_rgba(15,23,42,0.26)] dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(17,24,39,0.94)_0%,rgba(30,41,59,0.9)_100%)] dark:shadow-[0_12px_28px_-26px_rgba(2,6,23,0.74)]'
                                    }`}
                                >
                                    <div className="min-w-0 flex-1">
                                        <div className="flex items-center gap-3">
                                            <ProjectStatusDot active={project.running > 0} />
                                            <div
                                                className={`truncate text-[13px] ${
                                                    index === 0
                                                        ? 'font-semibold text-slate-900 dark:text-slate-100'
                                                        : 'font-medium text-foreground dark:text-slate-100'
                                                }`}
                                            >
                                                {project.project}
                                            </div>
                                        </div>
                                        <div
                                            className={`mt-0.5 text-[12px] ${
                                                index === 0 ? 'text-slate-500 dark:text-slate-400' : 'text-muted-foreground dark:text-slate-400'
                                            }`}
                                        >
                                            {t('home.landing.projects.item-summary', {
                                                total: project.total,
                                                running: project.running,
                                            })}
                                        </div>
                                    </div>
                                    <div
                                        className={`pl-3 ${isMobile ? 'text-[10px]' : 'text-[11px]'} ${
                                            index === 0 ? 'text-slate-500 dark:text-slate-400' : 'text-slate-400 dark:text-slate-500'
                                        }`}
                                    >
                                        {formatFriendlyDateTime(project.createdAt)}
                                    </div>
                                </div>
                            ))}

                            {!isLoading && projects.length === 0 ? (
                                <div className="rounded-2xl border border-dashed border-border bg-background px-4 py-10 text-center text-sm text-muted-foreground dark:border-white/10 dark:bg-slate-950/60 dark:text-slate-400">
                                    {t('home.landing.projects.empty')}
                                </div>
                            ) : null}
                        </div>
                    </div>

                    <div className="rounded-[22px] border border-white/85 bg-[linear-gradient(180deg,rgba(255,255,255,0.94)_0%,rgba(248,250,252,0.96)_100%)] p-3 shadow-[0_20px_52px_-36px_rgba(15,23,42,0.32)] dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.95)_0%,rgba(17,24,39,0.96)_100%)] dark:shadow-[0_20px_52px_-36px_rgba(2,6,23,0.75)] lg:p-2.5 xl:p-3.5 2xl:p-4">
                        <div className="text-[15px] font-semibold text-foreground dark:text-slate-100">
                            {t('home.landing.overview.title')}
                        </div>                        

                        <div className={`${isMobile ? 'mt-1.5 h-[120px]' : 'mt-1 h-[90px] lg:h-[82px] xl:mt-1.5 xl:h-[112px] 2xl:h-[124px]'} rounded-[18px] border border-slate-200/75 bg-[linear-gradient(180deg,rgba(248,250,252,0.96)_0%,rgba(255,255,255,0.96)_100%)] p-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.92)_0%,rgba(17,24,39,0.96)_100%)] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]`}>
                            {chartData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="2 10" vertical={false} stroke={chartGridStroke} />
                                        <YAxis
                                            type="number"
                                            allowDecimals={false}
                                            width={yAxisWidth}
                                            axisLine={false}
                                            tickLine={false}
                                            ticks={safeTicks}
                                            tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                                        />
                                        <XAxis
                                            dataKey="month"
                                            axisLine={false}
                                            tickLine={false}
                                            tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                                            tickFormatter={formatMonthLabel}
                                        />
                                        <Tooltip
                                            cursor={{ fill: chartCursorFill }}
                                            contentStyle={tooltipContentStyle}
                                            labelFormatter={(label) =>
                                                t('home.landing.chart.month-label', {
                                                    label: formatMonthLabel(String(label)),
                                                })
                                            }
                                            formatter={(value) => [
                                                `${value}`,
                                                t('home.landing.chart.run-count'),
                                            ]}
                                        />
                                        <Bar
                                            dataKey="count"
                                            radius={[10, 10, 0, 0]}
                                            fill="#2c8fe6"
                                        />
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                                    {t('home.landing.chart.empty')}
                                </div>
                            )}
                        </div>

                        <div className="mt-1 grid gap-1 lg:mt-0.5 lg:gap-0.5 xl:mt-1.5 xl:gap-1.5 2xl:mt-2 2xl:gap-2">
                            <div className="rounded-[16px] border border-slate-200/75 bg-[linear-gradient(180deg,rgba(248,250,252,0.92)_0%,rgba(241,245,249,0.92)_100%)] p-1.5 dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(17,24,39,0.92)_0%,rgba(30,41,59,0.88)_100%)]">
                                <div className="text-[11px] text-muted-foreground dark:text-slate-400">
                                    {t('home.landing.recent-project.title')}
                                </div>
                                <div className="mt-0.5 text-[14px] font-semibold text-foreground dark:text-slate-100">
                                    {projects[0]?.project ?? '--'}
                                </div>
                                <div className="mt-0.5 text-[11px] text-muted-foreground dark:text-slate-400">
                                    {projects[0]
                                        ? t('home.landing.recent-project.summary', {
                                              total: projects[0].total,
                                          })
                                        : t('home.landing.recent-project.empty')}
                                </div>
                            </div>

                            <div className="rounded-[16px] border border-slate-200/75 bg-[linear-gradient(180deg,rgba(248,250,252,0.92)_0%,rgba(241,245,249,0.92)_100%)] p-1.5 dark:border-white/8 dark:bg-[linear-gradient(180deg,rgba(17,24,39,0.92)_0%,rgba(30,41,59,0.88)_100%)]">
                                <div className="text-[11px] text-muted-foreground dark:text-slate-400">
                                    {t('home.landing.last-updated.title')}
                                </div>
                                <div className="mt-0.5 text-[14px] font-semibold text-foreground dark:text-slate-100">
                                    {formatFriendlyDateTime(latestRun?.timestamp)}
                                </div>
                                <div className="mt-0.5 text-[11px] text-muted-foreground dark:text-slate-400">
                                    {latestRun
                                        ? t('home.landing.last-updated.summary', {
                                             project: latestRun.project,
                                              status: latestRun.status,
                                          })
                                        : t('home.landing.last-updated.empty')}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                )}
            </div>
            <Modal
                title={null}
                open={isOnboardingOpen}
                onCancel={() => setIsOnboardingOpen(false)}
                footer={null}
                centered
                width={720}
            >
                <div className="px-2 py-2">
                    <div className="mb-5">
                        <h2 className="text-2xl font-semibold tracking-tight text-foreground">
                            {t('home.landing.getStarted.title')}
                        </h2>
                        {t('home.landing.getStarted.description') ? (
                            <p className="mt-2 text-sm leading-6 text-muted-foreground">
                                {t('home.landing.getStarted.description')}
                            </p>
                        ) : null}
                    </div>

                    <div className="grid gap-3 md:grid-cols-3">
                        <button
                            type="button"
                            onClick={handleStartPageTour}
                            className="rounded-[18px] border border-sky-100 bg-sky-50/70 p-4 text-left transition hover:-translate-y-0.5 hover:border-sky-200 hover:bg-sky-50 dark:border-sky-400/20 dark:bg-sky-500/10"
                        >
                            <Compass className="mb-3 h-6 w-6 text-sky-600" />
                            <div className="font-semibold text-foreground">
                                {t('home.landing.getStarted.pageTour.title')}
                            </div>
                            <div className="mt-1 text-xs leading-5 text-muted-foreground">
                                {t('home.landing.getStarted.pageTour.description')}
                            </div>
                        </button>

                        <button
                            type="button"
                            onClick={handleStartTrainingGuide}
                            className="rounded-[18px] border border-emerald-100 bg-emerald-50/70 p-4 text-left transition hover:-translate-y-0.5 hover:border-emerald-200 hover:bg-emerald-50 dark:border-emerald-400/20 dark:bg-emerald-500/10"
                        >
                            <BookOpen className="mb-3 h-6 w-6 text-emerald-600" />
                            <div className="font-semibold text-foreground">
                                {t('home.landing.getStarted.textGuide.title')}
                            </div>
                            <div className="mt-1 text-xs leading-5 text-muted-foreground">
                                {t('home.landing.getStarted.textGuide.description')}
                            </div>
                        </button>

                        <button
                            type="button"
                            onClick={handleStartPractice}
                            className="rounded-[18px] border border-orange-100 bg-orange-50/70 p-4 text-left transition hover:-translate-y-0.5 hover:border-orange-200 hover:bg-orange-50 dark:border-orange-400/20 dark:bg-orange-500/10"
                        >
                            <Rocket className="mb-3 h-6 w-6 text-orange-600" />
                            <div className="font-semibold text-foreground">
                                {t('home.landing.getStarted.practice.title')}
                            </div>
                            <div className="mt-1 text-xs leading-5 text-muted-foreground">
                                {t('home.landing.getStarted.practice.description')}
                            </div>
                        </button>
                    </div>

                    <div className="mt-5 flex flex-col gap-2 border-t border-border pt-4 sm:flex-row sm:items-center sm:justify-between">
                        <div className="text-xs text-muted-foreground">
                            {t('home.landing.getStarted.reminder')}
                        </div>
                        <Button onClick={handleContinueWithoutGuide}>
                            {t('home.landing.getStarted.continue')}
                        </Button>
                    </div>
                </div>
            </Modal>
        </div>
    );
};

export default LandingPage;
