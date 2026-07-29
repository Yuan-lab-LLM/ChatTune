import { memo, useMemo } from 'react';
import { Clock3 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ManagementCacheMeta } from '@shared/types';

interface Props {
    meta?: ManagementCacheMeta | null;
}

const CacheMetaInfo = ({ meta }: Props) => {
    const { t, i18n } = useTranslation();

    const formattedUpdatedAt = useMemo(() => {
        if (!meta?.updatedAt) {
            return t('query.refresh-meta.empty-time') || '暂无更新记录';
        }

        return new Intl.DateTimeFormat(i18n.language === 'zh' ? 'zh-CN' : 'en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
        }).format(new Date(meta.updatedAt));
    }, [i18n.language, meta?.updatedAt, t]);

    if (!meta) return null;

    return (
        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/60 px-3 py-1.5 text-muted-foreground">
                <Clock3 className="h-3.5 w-3.5" />
                <span>{t('query.refresh-meta.updated-at') || '更新时间'}</span>
                <span className="font-medium text-foreground">{formattedUpdatedAt}</span>
            </div>
        </div>
    );
};

export default memo(CacheMetaInfo);
