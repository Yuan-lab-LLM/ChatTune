import { useMemo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, Card, Tag, Spin, Alert } from 'antd';
import { Database, FileSearch, Upload, Eye, CheckCircle, Info, Search } from 'lucide-react';
import { LoadingOutlined } from '@ant-design/icons';
import type { DatasetInfo } from '@shared/types/trpc';

interface Step1DatasetIntroProps {
  queryState: 'idle' | 'querying' | 'completed';
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  onSelectDataset: (dataset: DatasetInfo) => void;
  requiredDatasetId?: string;
}

const REQUIRED_DATASET_NAME = 'medical-example';

export function Step1DatasetIntro({
  queryState,
  datasets,
  selectedDataset,
  onSelectDataset,
}: Step1DatasetIntroProps) {
  const { t } = useTranslation();

  // 检查是否有缓存数据
  const [cachedData, setCachedData] = useState<DatasetInfo[]>([]);
  const [hasCache, setHasCache] = useState(false);
  
  useEffect(() => {
    const cached = localStorage.getItem('cached_datasets');
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setCachedData(parsed);
          const hasExample = parsed.some((d: DatasetInfo) => d.name === REQUIRED_DATASET_NAME);
          setHasCache(hasExample);
        }
      } catch (e) {
        console.error('Failed to parse cached datasets:', e);
      }
    }
  }, []);
  
  // 优先使用实际查询的数据，如果没有则使用缓存
  const displayDatasets = datasets.length > 0 ? datasets : cachedData;
  const effectiveQueryState = datasets.length > 0 ? queryState : (hasCache ? 'completed' : queryState);
  
  // 查找示例数据集
  const exampleDataset = useMemo(() => {
    return displayDatasets.find(d => d.name === REQUIRED_DATASET_NAME);
  }, [displayDatasets]);

  // 其他数据集
  const otherDatasets = useMemo(() => {
    return displayDatasets.filter(d => d.name !== REQUIRED_DATASET_NAME);
  }, [displayDatasets]);

  // 判断是否可以选择示例数据集
  const canSelectExample = effectiveQueryState === 'completed' && exampleDataset;
  const hasSelectedExample = selectedDataset?.name === REQUIRED_DATASET_NAME;

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        <div className="p-3 bg-blue-100 rounded-2xl shrink-0 border border-blue-200/60">
          <Database className="w-8 h-8 text-blue-600" />
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-semibold mb-2">
            {effectiveQueryState === 'idle' && !hasCache && t('wizard.step1.title')}
            {effectiveQueryState === 'idle' && hasCache && t('wizard.step1.title')}
            {effectiveQueryState === 'querying' && t('wizard.step1.querying')}
            {effectiveQueryState === 'completed' && t('wizard.step1.title')}
          </h4>
          <p className="text-muted-foreground">
            {effectiveQueryState === 'idle' && !hasCache && t('wizard.step1.guide.desc')}
            {effectiveQueryState === 'idle' && hasCache && t('wizard.step1.cacheFound')}
            {effectiveQueryState === 'querying' && t('wizard.step1.queryingDesc')}
            {effectiveQueryState === 'completed' && t('wizard.step1.queryComplete')}
          </p>
        </div>
      </div>

      {/* 查询引导 - 只在 idle 状态且没有缓存时显示 */}
      {effectiveQueryState === 'idle' && !hasCache && (
        <Card 
          className="rounded-2xl border-amber-200/70 shadow-none bg-amber-50/70"
          size="small"
        >
          <div className="flex items-start gap-4">
            <div className="p-3 bg-amber-100 rounded-2xl shrink-0 border border-amber-200/70">
              <Search className="w-8 h-8 text-amber-600" />
            </div>
            <div className="flex-1">
              <h5 className="font-semibold text-amber-900 mb-3">💡 {t('wizard.step1.guide.title')}</h5>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-amber-500 text-white flex items-center justify-center font-bold text-sm">1</div>
                  <div>
                    <div className="font-medium text-gray-900">{t('wizard.step1.guide.step1')}</div>
                    <div className="text-sm text-gray-600">{t('wizard.step1.guide.step1Desc')}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-amber-500 text-white flex items-center justify-center font-bold text-sm">2</div>
                  <div>
                    <div className="font-medium text-gray-900">{t('wizard.step1.guide.step2')}</div>
                    <div className="text-sm text-gray-600">{t('wizard.step1.guide.step2Desc')}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-amber-500 text-white flex items-center justify-center font-bold text-sm">3</div>
                  <div>
                    <div className="font-medium text-gray-900">{t('wizard.step1.guide.step3')}</div>
                    <div className="text-sm text-gray-600">{t('wizard.step1.guide.step3Desc')}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <Alert
            message={`⚠️ ${t('wizard.step1.selectRequired')}`}
            description={t('wizard.step1.selectRequiredDesc')}
            type="warning"
            showIcon
            className="mt-4 rounded-2xl border-amber-200/70 bg-white/80"
          />
        </Card>
      )}

      {/* 查询中状态 */}
      {effectiveQueryState === 'querying' && (
        <Card className="rounded-2xl border-blue-200/70 shadow-none bg-blue-50/70" size="small">
          <div className="flex flex-col items-center justify-center py-8">
            <Spin indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />} />
            <div className="mt-4 text-center">
              <div className="font-semibold text-blue-900">{t('wizard.step1.querying')}</div>
              <div className="text-sm text-blue-600 mt-1">{t('wizard.step1.queryingDesc')}</div>
            </div>
          </div>
        </Card>
      )}

      {/* 查询完成后的数据集列表 - 包括缓存数据 */}
      {(effectiveQueryState === 'completed' || (effectiveQueryState === 'idle' && hasCache)) && (
        <div>
          {/* 示例数据集 */}
          {exampleDataset && (
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <h5 className="font-semibold text-green-800">
                  {hasCache && effectiveQueryState === 'idle'
                    ? `✅ ${t('wizard.step1.useCache')}`
                    : `✅ ${t('wizard.step1.foundExample')}`}
                </h5>
              </div>

              {hasCache && effectiveQueryState === 'idle' && (
                <Alert
                  message={t('wizard.step1.useCache')}
                  description={t('wizard.step1.cacheDesc')}
                  type="info"
                  showIcon
                  className="mb-3 rounded-2xl border-border/25 bg-white/80"
                />
              )}
              
              <Card
                className={`cursor-pointer transition-all ${
                  hasSelectedExample
                    ? 'border-2 border-green-500 bg-green-50 rounded-2xl'
                    : 'border-2 border-green-300 bg-green-50/50 rounded-2xl hover:border-green-500 hover:shadow-sm'
                }`}
                onClick={() => onSelectDataset(exampleDataset)}
                size="small"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-2xl">📄</span>
                      <span className="font-semibold text-lg text-green-900">{exampleDataset.name}</span>
                      <Tag color="green">{t('wizard.step1.exampleData')}</Tag>
                      <Tag color="blue">{t('wizard.step1.recommended')}</Tag>
                      {hasSelectedExample && <CheckCircle className="w-5 h-5 text-green-600" />}
                    </div>
                    
                    <p className="text-gray-600 text-sm mb-2">
                      {t('wizard.step1.exampleDesc')}
                    </p>

                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>{t('wizard.step1.type')}: <Tag size="small">{exampleDataset.format || 'RAW'}</Tag></span>
                      {exampleDataset.recordCount && (
                        <span>{t('wizard.step1.recordCount')}: <Tag size="small">{exampleDataset.recordCount.toLocaleString()}{t('common.records') || '条'}</Tag></span>
                      )}
                    </div>
                  </div>

                  {hasSelectedExample ? (
                    <Button type="primary" size="small" className="ml-4 h-9 rounded-xl bg-green-600">{t('wizard.step1.selected')}</Button>
                  ) : (
                    <Button type="primary" size="small" className="ml-4 h-9 rounded-xl">{t('wizard.step1.select')}</Button>
                  )}
                </div>
              </Card>
            </div>
          )}

          {/* 没有找到示例数据集 */}
          {!exampleDataset && (
            <Alert
              message={t('wizard.step1.notFound')}
              description={
                <div>
                  <p>{t('wizard.step1.notFoundDesc')}</p>
                  <p className="mt-2">{t('wizard.step1.reasons.title')}</p>
                  <ul className="list-disc list-inside mt-1">
                    <li>{t('wizard.step1.reasons.reason1')}</li>
                    <li>{t('wizard.step1.reasons.reason2')}</li>
                  </ul>
                  <p className="mt-2 text-amber-600">{t('wizard.step1.reasons.contactAdmin')}</p>
                </div>
              }
              type="warning"
              showIcon
              className="rounded-2xl border-amber-200/70 bg-amber-50/70"
            />
          )}

          {/* 其他数据集（如果有） */}
          {otherDatasets.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Info className="w-5 h-5 text-gray-500" />
                <h5 className="font-semibold text-gray-600">{t('wizard.step1.otherDatasets')}</h5>
              </div>

              <div className="space-y-2">
                {otherDatasets.slice(0, 3).map(dataset => (
                  <Card
                    key={dataset.name}
                    className="bg-muted/20 border-border/25 rounded-2xl shadow-none cursor-not-allowed opacity-60"
                    size="small"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xl">📁</span>
                        <span className="font-medium text-gray-500">{dataset.name}</span>
                      </div>
                      <Tag color="default" className="text-xs">{t('wizard.step1.unavailable')}</Tag>
                    </div>
                  </Card>
                ))}
                {otherDatasets.length > 3 && (
                  <div className="text-center text-sm text-gray-500 py-2">
                    {t('wizard.step1.moreDatasets', { count: otherDatasets.length - 3 })}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 提示信息 */}
          {!hasSelectedExample && exampleDataset && (
            <Alert
              message={t('wizard.step1.mustSelectExample')}
              description={t('wizard.step1.mustSelectExampleDesc')}
              type="warning"
              showIcon
              className="mt-4"
            />
          )}

          {hasSelectedExample && (
            <Alert
              message={t('wizard.step1.selectionStatus', { name: selectedDataset?.name })}
              description={t('wizard.step1.selectionStatusDesc')}
              type="success"
              showIcon
              className="mt-4"
            />
          )}
        </div>
      )}
    </div>
  );
}
