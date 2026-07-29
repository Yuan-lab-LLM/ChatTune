import { memo, ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowLeft, MousePointerClick } from 'lucide-react';

import WipIcon from '@/assets/svgs/page-wip.svg?react';
import EmptyIcon from '@/assets/svgs/page-empty.svg?react';
import NotFoundIcon from '@/assets/svgs/page-notFound.svg?react';

interface Props {
    icon: ReactNode;
    title: string;
}

const BaseDefaultPage = ({ icon, title }: Props) => {
    return (
        <div className="flex flex-col items-center justify-center h-full w-full">
            {icon}
            <div className="text-primary/40 text-sm">{title}</div>
        </div>
    );
};

export const ProjectNotFoundPage = memo(() => {
    const { t } = useTranslation();

    return (
        <BaseDefaultPage
            icon={<NotFoundIcon width={350} height={350} />}
            title={t('error.project-not-found')}
        />
    );
});

export const RunNotFoundPage = memo(() => {
    const { t } = useTranslation();

    return (
        <BaseDefaultPage
            icon={<NotFoundIcon width={350} height={350} />}
            title={t('error.run-not-found')}
        />
    );
});

export const EmptyRunPage = memo(() => {
    const { t } = useTranslation();

    return (
        <div className="flex h-full w-full items-center justify-center px-6">
            <div className="max-w-[620px] rounded-[32px] border border-slate-200/70 bg-white/88 px-10 py-12 text-center shadow-[0_28px_80px_-48px_rgba(15,23,42,0.38)] backdrop-blur-sm">
                <div className="mx-auto mb-6 flex h-[220px] items-center justify-center overflow-hidden rounded-[28px] bg-slate-50/80">
                    <EmptyIcon width={260} height={260} />
                </div>
                <div className="mx-auto mb-3 inline-flex items-center gap-2 rounded-full border border-blue-200/70 bg-blue-50 px-4 py-2 text-[13px] font-medium text-blue-700">
                    <MousePointerClick className="h-4 w-4" />
                    {t('hint.select-run-badge')}
                </div>
                <div className="mb-3 text-[28px] font-semibold tracking-tight text-slate-900">
                    {t('hint.select-run')}
                </div>
                <div className="mx-auto max-w-[440px] text-[15px] leading-7 text-slate-500">
                    {t('hint.select-run-desc')}
                </div>
                <div className="mx-auto mt-6 inline-flex items-center gap-2 rounded-full bg-slate-900 px-4 py-2 text-[13px] font-medium text-white shadow-[0_16px_28px_-20px_rgba(15,23,42,0.65)]">
                    <ArrowLeft className="h-4 w-4" />
                    {t('hint.select-run-action')}
                </div>
            </div>
        </div>
    );
});

export const EmptyMessagePage = memo(() => {
    const { t } = useTranslation();

    return (
        <BaseDefaultPage
            icon={<EmptyIcon width={150} height={150} />}
            title={t('hint.select-message')}
        />
    );
});


export const WipPage = memo(() => {
    return (
        <BaseDefaultPage
            icon={<WipIcon width={250} height={250} />}
            title="Coming soon ..."
        />
    );
});

export const EmptyPage = memo(
    ({ size, title }: { size: number; title: string }) => {
        return (
            <BaseDefaultPage
                icon={<EmptyIcon width={size} height={size} />}
                title={title}
            />
        );
    },
);
