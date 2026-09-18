import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Alert, Button, Card, Spin, Tag } from 'antd';
import { CheckCircle, Database, Info, Search } from 'lucide-react';
import { LoadingOutlined } from '@ant-design/icons';
import type { DatasetInfo } from '@shared/types/trpc';
import {
  DEFAULT_PREFERRED_DATASET_ID,
  isMedicalDemoDataset,
  isRawDataset,
} from '../config';

interface Step1DatasetIntroProps {
  queryState: 'idle' | 'querying' | 'completed';
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  onSelectDataset: (dataset: DatasetInfo) => void;
  requiredDatasetId?: string;
}

export function Step1DatasetIntro({
  queryState,
  datasets,
  selectedDataset,
  onSelectDataset,
  requiredDatasetId = DEFAULT_PREFERRED_DATASET_ID,
}: Step1DatasetIntroProps) {
  const { t } = useTranslation();
  const [cachedData, setCachedData] = useState<DatasetInfo[]>([]);

  useEffect(() => {
    const cached = localStorage.getItem('cached_datasets');
    if (!cached) return;
    try {
      const parsed = JSON.parse(cached);
      if (Array.isArray(parsed)) setCachedData(parsed);
    } catch (error) {
      console.error('Failed to parse cached datasets:', error);
    }
  }, []);

  const hasRawCache = cachedData.some(isRawDataset);
  const displayDatasets = datasets.length > 0 ? datasets : cachedData;
  const effectiveQueryState = datasets.length > 0
    ? queryState
    : (hasRawCache ? 'completed' : queryState);

  const rawDatasets = useMemo(() => {
    const items = displayDatasets.filter(isRawDataset);
    return [...items].sort((left, right) => {
      const score = (dataset: DatasetInfo) => {
        if (dataset.name === requiredDatasetId) return 0;
        if (!isMedicalDemoDataset(dataset)) return 1;
        return 2;
      };
      return score(left) - score(right) || left.name.localeCompare(right.name);
    });
  }, [displayDatasets, requiredDatasetId]);

  const recommendedDataset = rawDatasets[0];
  const nonRawDatasets = displayDatasets.filter((dataset) => !isRawDataset(dataset));
  const hasSelectedRaw = Boolean(selectedDataset && isRawDataset(selectedDataset));

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div className="p-3 bg-blue-100 rounded-2xl shrink-0 border border-blue-200/60">
          <Database className="w-8 h-8 text-blue-600" />
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-semibold mb-2">
            {effectiveQueryState === 'querying' ? t('wizard.step1.querying') : t('wizard.step1.title')}
          </h4>
          <p className="text-muted-foreground">
            {effectiveQueryState === 'querying'
              ? t('wizard.step1.queryingDesc')
              : effectiveQueryState === 'completed'
                ? t('wizard.step1.queryComplete')
                : t('wizard.step1.guide.desc')}
          </p>
        </div>
      </div>

      {effectiveQueryState === 'idle' && !hasRawCache && (
        <Card className="rounded-2xl border-amber-200/70 shadow-none bg-amber-50/70" size="small">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-amber-100 rounded-2xl shrink-0 border border-amber-200/70">
              <Search className="w-8 h-8 text-amber-600" />
            </div>
            <div className="flex-1">
              <h5 className="font-semibold text-amber-900 mb-3">{t('wizard.step1.guide.title')}</h5>
              <div className="space-y-2 text-sm text-gray-700">
                <div>1. {t('wizard.step1.guide.step1')} - {t('wizard.step1.guide.step1Desc')}</div>
                <div>2. {t('wizard.step1.guide.step2')} - {t('wizard.step1.guide.step2Desc')}</div>
                <div>3. {t('wizard.step1.guide.step3')} - {t('wizard.step1.guide.step3Desc')}</div>
              </div>
            </div>
          </div>
          <Alert
            message={t('wizard.step1.selectRequired')}
            description={t('wizard.step1.selectRequiredDesc')}
            type="warning"
            showIcon
            className="mt-4 rounded-2xl border-amber-200/70 bg-white/80"
          />
        </Card>
      )}

      {effectiveQueryState === 'querying' && (
        <Card className="rounded-2xl border-blue-200/70 shadow-none bg-blue-50/70" size="small">
          <div className="flex flex-col items-center justify-center py-8">
            <Spin indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />} />
            <div className="mt-4 text-sm text-blue-700">{t('wizard.step1.queryingDesc')}</div>
          </div>
        </Card>
      )}

      {effectiveQueryState === 'completed' && rawDatasets.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <h5 className="font-semibold text-green-800">{t('wizard.step1.foundExample')}</h5>
          </div>
          {rawDatasets.map((dataset) => {
            const selected = selectedDataset?.name === dataset.name;
            const recommended = dataset.name === recommendedDataset?.name;
            return (
              <Card
                key={`${dataset.containerName || ''}:${dataset.path || dataset.name}`}
                className={`cursor-pointer rounded-2xl transition-all ${
                  selected
                    ? 'border-2 border-green-500 bg-green-50'
                    : 'border-2 border-border/30 bg-background hover:border-blue-400 hover:shadow-sm'
                }`}
                onClick={() => onSelectDataset(dataset)}
                size="small"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 mb-2">
                      <span className="font-semibold text-base">{dataset.name}</span>
                      <Tag color="blue">RAW</Tag>
                      {recommended && <Tag color="green">{t('wizard.step1.recommended')}</Tag>}
                      {isMedicalDemoDataset(dataset) && <Tag>{t('wizard.step1.medicalExample')}</Tag>}
                    </div>
                    <p className="text-sm text-muted-foreground break-all">
                      {dataset.description || dataset.path || t('wizard.step1.genericDesc')}
                    </p>
                  </div>
                  <Button type={selected ? 'primary' : 'default'} size="small">
                    {selected ? t('wizard.step1.selected') : t('wizard.step1.select')}
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {effectiveQueryState === 'completed' && rawDatasets.length === 0 && (
        <Alert
          message={t('wizard.step1.notFound')}
          description={t('wizard.step1.notFoundDesc')}
          type="warning"
          showIcon
          className="rounded-2xl border-amber-200/70 bg-amber-50/70"
        />
      )}

      {effectiveQueryState === 'completed' && nonRawDatasets.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-2 text-muted-foreground">
            <Info className="w-4 h-4" />
            <span className="text-sm font-medium">{t('wizard.step1.nonRawDatasets')}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {nonRawDatasets.slice(0, 6).map((dataset) => (
              <Tag key={`${dataset.type}:${dataset.name}`}>{dataset.name} ({dataset.type})</Tag>
            ))}
          </div>
        </div>
      )}

      {hasSelectedRaw && (
        <Alert
          message={t('wizard.step1.selectionStatus', { name: selectedDataset?.name })}
          description={t('wizard.step1.selectionStatusDesc')}
          type="success"
          showIcon
        />
      )}
    </div>
  );
}
