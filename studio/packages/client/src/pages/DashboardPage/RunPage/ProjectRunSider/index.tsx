import { Empty, Flex, Select } from "antd";
import { Key, memo, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useParams } from "react-router-dom";
import { HelpCircle } from "lucide-react";

import AsTable from "@/components/tables/AsTable";

import { StatusCell, TextCell } from "@/components/tables/utils.tsx";
import { useProjectRoom } from "@/context/ProjectRoomContext.tsx";
import { useTour } from "@/context/TourContext.tsx";
import { RemoveScrollBarStyle } from "@/styles.ts";
import {
  Status,
  SystemOverviewData,
  GPUInfo,
  DatasetInfo,
  ModelInfo,
  MedicalTestFile,
  EvaluationResult,
  ManagementCacheMeta,
} from "@shared/types";
import SystemOverviewSider from "../SystemOverviewSider";
import DatasetManager from "../DatasetManager";
import ModelManager from "../ModelManager";
import EvaluationManager from "../EvaluationManager";
import { GuideModal } from "@/components/GuideModal";
import { useIsMobile } from "@/hooks/use-mobile";
import { useTheme } from "@/context/ThemeContext.tsx";
import {
  ResourceNodeSelector,
  useResourceNodeSelection,
} from "@/hooks/useResourceNodeSelection";
import { useAuth } from "@/context/AuthContext";
import "./index.css";

/**
 * Sider width configurations for folded and unfolded states.
 */
enum SiderDrawerWidth {
  UNFOLDED = "80vw",
  FOLDED = 200,
}

/**
 * Props for the project run sidebar component.
 */
interface Props {
  onRunClick: (runId: string) => void;
  systemOverviewData: SystemOverviewData | null;
  gpuInfo: GPUInfo[] | null;
  onRefreshGPUInfo?: () => void;
  activeTab: "runs" | "overview" | "datasets" | "models" | "evaluation";
  onTabChange: (
    tab: "runs" | "overview" | "datasets" | "models" | "evaluation",
  ) => void;
  onClose?: () => void;
  // Dataset management props
  datasets: DatasetInfo[];
  isQueryingDatasets: boolean;
  datasetQueryError: boolean;
  datasetErrorMessage?: string;
  hasQueriedDatasets: boolean;
  onQueryDatasets: (queryText: string) => void;
  onRefreshDatasets?: (queryText: string) => void;
  datasetCacheMeta?: ManagementCacheMeta | null;
  onUpload?: () => void;
  onDownload?: (dataset: DatasetInfo) => void;
  onDeleteDataset?: (dataset: DatasetInfo) => Promise<void>;
  onUseDatasetForTraining?: (dataset: DatasetInfo) => void;
  onLoadDatasetPreviews?: (dataset: DatasetInfo) => Promise<void>;
  isInputDisabled?: boolean;
  inputDisabledHint?: string;
  isUploading?: boolean;
  downloadingId?: string | null;
  deletingDatasetId?: string | null;
  // Model management props
  models: ModelInfo[];
  isQueryingModels: boolean;
  modelQueryError: boolean;
  modelErrorMessage?: string;
  hasQueriedModels: boolean;
  onQueryModels: (queryText: string) => void;
  onRefreshModels?: (queryText: string) => void;
  modelCacheMeta?: ManagementCacheMeta | null;
  onDeleteModel?: (model: ModelInfo) => Promise<void>;
  deletingModelId?: string | null;
  // Evaluation management props
  tests: MedicalTestFile[];
  isQueryingTests: boolean;
  testQueryError: boolean;
  testErrorMessage?: string;
  hasQueriedTests: boolean;
  onQueryTests: (containerName: string) => void;
  onRefreshTests?: (containerName: string) => void;
  testCacheMeta?: ManagementCacheMeta | null;
  onUploadTest?: () => void;
  onDownloadTest?: (testName: string, test?: MedicalTestFile) => void;
  onDeleteTest?: (testName: string, test?: MedicalTestFile) => Promise<void>;
  onUseEvaluationForBenchmark?: (testName: string) => void;
  isUploadingTest?: boolean;
  downloadingTestId?: string | null;
  deletingTestId?: string | null;
  // Evaluation results props
  evaluationResults?: EvaluationResult[];
  isQueryingEvaluationResults?: boolean;
  evaluationResultQueryError?: boolean;
  evaluationResultErrorMessage?: string;
  hasQueriedEvaluationResults?: boolean;
  onQueryEvaluationResults?: (containerName: string) => void;
  onRefreshEvaluationResults?: (containerName: string) => void;
  evaluationResultCacheMeta?: ManagementCacheMeta | null;
  onDownloadEvaluationResult?: (
    folderPath: string,
    filename: string,
    result?: EvaluationResult,
  ) => void;
  downloadingResultId?: string | null;
  onDeleteEvaluationResult?: (
    folderPath: string,
    result?: EvaluationResult,
  ) => Promise<void>;
  deletingResultId?: string | null;
  currentRunNodeId?: string;
  currentTrainingContainerName?: string;
  currentEvaluationContainerName?: string;
  // focusOnLatestRun for auto-selecting latest run
  focusOnLatestRun?: boolean;
}

/**
 * Sidebar component for displaying and managing project runs.
 * Features run table, search, auto-focus on latest run, and tour integration.
 */
const ProjectRunSider = ({
  onRunClick,
  systemOverviewData,
  gpuInfo,
  onRefreshGPUInfo,
  activeTab,
  onTabChange,
  datasets,
  isQueryingDatasets,
  datasetQueryError,
  datasetErrorMessage,
  hasQueriedDatasets,
  onQueryDatasets,
  onRefreshDatasets,
  datasetCacheMeta,
  onUpload,
  onDownload,
  onDeleteDataset,
  onUseDatasetForTraining,
  onLoadDatasetPreviews,
  isInputDisabled,
  inputDisabledHint,
  isUploading,
  downloadingId,
  deletingDatasetId,
  models,
  isQueryingModels,
  modelQueryError,
  modelErrorMessage,
  hasQueriedModels,
  onQueryModels,
  onRefreshModels,
  modelCacheMeta,
  onDeleteModel,
  deletingModelId,
  tests,
  isQueryingTests,
  testQueryError,
  testErrorMessage,
  hasQueriedTests,
  onQueryTests,
  onRefreshTests,
  testCacheMeta,
  onUploadTest,
  onDownloadTest,
  onDeleteTest,
  onUseEvaluationForBenchmark,
  isUploadingTest,
  downloadingTestId,
  deletingTestId,
  // Evaluation results props
  evaluationResults = [],
  isQueryingEvaluationResults = false,
  evaluationResultQueryError = false,
  evaluationResultErrorMessage = "",
  hasQueriedEvaluationResults = false,
  onQueryEvaluationResults,
  onRefreshEvaluationResults,
  evaluationResultCacheMeta,
  onDownloadEvaluationResult,
  downloadingResultId,
  onDeleteEvaluationResult,
  deletingResultId,
  currentRunNodeId,
  currentTrainingContainerName,
  currentEvaluationContainerName,
  focusOnLatestRun,
}: Props) => {
  const { nodeId, nodes } = useResourceNodeSelection();
  const { isAdmin } = useAuth();
  const isAllNodes = nodeId === "all";
  const isMobile = useIsMobile();
  const { resolvedTheme } = useTheme();
  const { t } = useTranslation();
  const guideContent = [
    `1. ${t("runpage.guide.section1.title") || "页面布局说明"}`,
    `  - ${t("runpage.guide.section1.item1") || "点击左侧导航按钮可打开或收起管理面板"}`,
    `  - ${t("runpage.guide.section1.item2") || "右侧主区域用于对话、训练和推理操作"}`,
    "",
    `2. ${t("runpage.guide.section2.title") || "功能管理区域"}`,
    `  ${t("runpage.guide.section2.group1.title") || "运行列表"}`,
    `    - ${t("runpage.guide.section2.group1.item1") || "点击运行中的实例即可开启对话页面"}`,
    `  ${t("runpage.guide.section2.group2.title") || "系统概览"}`,
    `    - ${t("runpage.guide.section2.group2.item1") || "实时监控系统状态"}`,
    `    - ${t("runpage.guide.section2.group2.item2") || "查看 GPU 使用情况"}`,
    `  ${t("runpage.guide.section2.group3.title") || "数据管理"}`,
    `    - ${t("runpage.guide.section2.group3.item1") || "查询容器中的可用数据集"}`,
    `    - ${t("runpage.guide.section2.group3.item2") || "支持上传及下载"}`,
    `    - ${t("runpage.guide.section2.group3.item3") || "查看数据集预览（前3条数据）"}`,
    `  ${t("runpage.guide.section2.group4.title") || "模型管理"}`,
    `    - ${t("runpage.guide.section2.group4.item1") || "查询容器中的可用模型"}`,
    `  ${t("runpage.guide.section2.group5.title") || "评测管理"}`,
    `    - ${t("runpage.guide.section2.group5.item1") || "查询容器中的可用评测集"}`,
    `    - ${t("runpage.guide.section2.group5.item2") || "支持上传及下载"}`,
    "",
    `3. ${t("runpage.guide.section3.title") || "对话交互区域"}`,
    `  - ${t("runpage.guide.section3.item1") || "自然语言交互：直接输入指令与 AI 对话"}`,
    `  - ${t("runpage.guide.section3.item2") || '模板库：点击底部"模板库"按钮快速选择预设指令'}`,
    `  - ${t("runpage.guide.section3.item3") || "点击“清空上下文”按钮以开启新对话"}`,
    `  - ${t("runpage.guide.section3.item4") || "支持多种格式的对话记录导出"}`,
    "",
    `4. ${t("runpage.guide.section4.title") || "训练监控面板"}`,
    `  - ${t("runpage.guide.section4.item1") || "训练开始后，从训练任务状态栏打开监控面板"}`,
    `  - ${t("runpage.guide.section4.item2") || "显示 Loss/LR 曲线"}`,
    `  - ${t("runpage.guide.section4.item3") || "支持多进程对比"}`,
    `  - ${t("runpage.guide.section4.item4") || "可询问AI关于训练情况"}`,
    "",
    `5. ${t("runpage.guide.section5.title") || "常见问题"}`,
    `  ${t("runpage.guide.section5.group1.title") || "为什么看不到数据/模型？"}`,
    `    - ${t("runpage.guide.section5.group1.item1") || "请确认容器名称是否正确"}`,
    `    - ${t("runpage.guide.section5.group1.item2") || "确认该容器中是否真的有数据/模型"}`,
    `    - ${t("runpage.guide.section5.group1.item3") || "大小为 0 B 的项目会被自动过滤"}`,
    `  ${t("runpage.guide.section5.group2.title") || "如何查看数据集内容？"}`,
    `    - ${t("runpage.guide.section5.group2.item1") || '点击数据集卡片的"展开"按钮'}`,
    `    - ${t("runpage.guide.section5.group2.item2") || "可预览前3条数据"}`,
    `  ${t("runpage.guide.section5.group3.title") || "评测如何使用？"}`,
    `    - ${t("runpage.guide.section5.group3.item1") || '点击左侧"评测管理"入口，在"开源评测"子区域中上传评测文件，通过对话启动开源评测'}`,
  ].join("\n");
  const { runs } = useProjectRoom();
  const { registerRunPageTourStep } = useTour();
  const refTable = useRef(null);

  const [folded] = useState<boolean>(false);
  const [selectedRowKeys] = useState<Key[]>([]);
  const [isRunPageGuideOpen, setIsRunPageGuideOpen] = useState<boolean>(false);
  const [datasetContainerFilter, setDatasetContainerFilter] = useState("all");
  const [modelContainerFilter, setModelContainerFilter] = useState("all");
  const [testContainerFilter, setTestContainerFilter] = useState("all");
  const buildContainerOptions = useMemo(
    () => (items: Array<{ containerName?: string }>) => {
      const containerNames = Array.from(
        new Set(
          items
            .map((item) => item.containerName?.trim())
            .filter((name): name is string => Boolean(name)),
        ),
      ).sort((a, b) => a.localeCompare(b));

      return [
        { label: t("tag.all-short") || "全部", value: "all" },
        ...containerNames.map((name) => ({ label: name, value: name })),
      ];
    },
    [t],
  );
  const datasetContainerOptions = useMemo(
    () => buildContainerOptions(datasets),
    [buildContainerOptions, datasets],
  );
  const modelContainerOptions = useMemo(
    () => buildContainerOptions(models),
    [buildContainerOptions, models],
  );
  const evaluationContainerOptions = useMemo(
    () => buildContainerOptions([...tests, ...evaluationResults]),
    [buildContainerOptions, evaluationResults, tests],
  );
  const filteredDatasets = useMemo(
    () =>
      datasetContainerFilter === "all"
        ? datasets
        : datasets.filter(
            (dataset) => dataset.containerName === datasetContainerFilter,
          ),
    [datasetContainerFilter, datasets],
  );
  const filteredModels = useMemo(
    () =>
      modelContainerFilter === "all"
        ? models
        : models.filter((model) => model.containerName === modelContainerFilter),
    [modelContainerFilter, models],
  );
  const filteredTests = useMemo(
    () =>
      testContainerFilter === "all"
        ? tests
        : tests.filter((test) => test.containerName === testContainerFilter),
    [testContainerFilter, tests],
  );
  const filteredEvaluationResults = useMemo(
    () =>
      testContainerFilter === "all"
        ? evaluationResults
        : evaluationResults.filter(
            (result) => result.containerName === testContainerFilter,
          ),
    [evaluationResults, testContainerFilter],
  );
  useEffect(() => {
    if (!datasetContainerOptions.some((option) => option.value === datasetContainerFilter)) {
      setDatasetContainerFilter("all");
    }
  }, [datasetContainerFilter, datasetContainerOptions]);
  useEffect(() => {
    if (!modelContainerOptions.some((option) => option.value === modelContainerFilter)) {
      setModelContainerFilter("all");
    }
  }, [modelContainerFilter, modelContainerOptions]);
  useEffect(() => {
    if (!evaluationContainerOptions.some((option) => option.value === testContainerFilter)) {
      setTestContainerFilter("all");
    }
  }, [evaluationContainerOptions, testContainerFilter]);
  const renderDockerFilter = () => {
    const config =
      activeTab === "datasets"
        ? {
            value: datasetContainerFilter,
            onChange: setDatasetContainerFilter,
            options: datasetContainerOptions,
          }
        : activeTab === "models"
          ? {
              value: modelContainerFilter,
              onChange: setModelContainerFilter,
              options: modelContainerOptions,
            }
          : activeTab === "evaluation"
            ? {
                value: testContainerFilter,
                onChange: setTestContainerFilter,
                options: evaluationContainerOptions,
              }
            : null;

    if (!isAdmin || !config || config.options.length <= 1) {
      return null;
    }

    return (
      <>
        <span className="shrink-0 text-[10px] font-medium text-muted-foreground">
          Docker
        </span>
        <Select
          size="small"
          value={config.value}
          options={config.options}
          onChange={config.onChange}
          popupMatchSelectWidth={false}
          className="w-[94px] flex-none [&_.ant-select-selector]:!h-7 [&_.ant-select-selector]:!rounded-lg [&_.ant-select-selector]:!px-2 [&_.ant-select-selection-item]:!text-[11px] [&_.ant-select-selection-item]:!font-medium [&_.ant-select-selection-item]:!leading-[26px] [&_.ant-select-arrow]:!text-[10px]"
          aria-label="Docker"
        />
      </>
    );
  };

  const panelTitle = useMemo(() => {
    switch (activeTab) {
      case "runs":
        return t("overview.tab.runs");
      case "overview":
        return t("overview.tab.overview");
      case "datasets":
        return t("tab.datasets") || "数据管理";
      case "models":
        return t("tab.models") || "模型管理";
      case "evaluation":
        return t("tab.evaluation") || "评测管理";
      default:
        return "";
    }
  }, [activeTab, t]);

  // Register tour step for the run table
  useEffect(() => {
    registerRunPageTourStep({
      title: t("tour.run.run-table-title"),
      description: t("tour.run.run-table-description"),
      target: refTable.current,
      placement: "right",
    });
  }, []);

  // Extract current run and project from URL
  const { projectName, runId } = useParams();
  const project = projectName;
  const siderBackground =
    resolvedTheme === "dark"
      ? "linear-gradient(180deg, rgba(15,23,42,0.92) 0%, rgba(17,24,39,0.96) 100%)"
      : "linear-gradient(180deg, rgba(255,255,255,0.86) 0%, rgba(248,250,252,0.94) 100%)";
  const siderInsetShadow =
    resolvedTheme === "dark"
      ? "inset -1px 0 0 rgba(148, 163, 184, 0.08)"
      : "inset -1px 0 0 rgba(148, 163, 184, 0.12)";
  const siderBorderRight =
    resolvedTheme === "dark"
      ? "1px solid rgba(148, 163, 184, 0.1)"
      : "1px solid rgba(148, 163, 184, 0.12)";

  // Auto-navigate to latest run when focus mode is enabled
  useEffect(() => {
    if (focusOnLatestRun && runs.length > 0) {
      const latestRun = runs.reduce((prev, current) => {
        return prev.timestamp > current.timestamp ? prev : current;
      });

      if (latestRun.id !== runId) {
        onRunClick(latestRun.id);
      }
    }
  }, [runs, focusOnLatestRun]);

  return (
    <div className="h-full overflow-hidden" style={{ zIndex: 1 }}>
      <Flex
        ref={refTable}
        className="animated-sider-content runpage-sider-compact"
        style={{
          width: "100%",
          minWidth: folded ? SiderDrawerWidth.FOLDED : isMobile ? 0 : 270,
          maxWidth: "100%",
          padding: isMobile ? 10 : 12,
          height: "100%",
          background: siderBackground,
          transition: "min-width 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          boxShadow: folded || isMobile ? "none" : siderInsetShadow,
          position: "relative",
          borderRight: isMobile ? "none" : siderBorderRight,
          overflow: "hidden",
        }}
        vertical={true}
        gap={0}
      >
        <div className="runpage-side-panel">
          <div className="runpage-side-panel__header">
            <div>
              <div className="runpage-side-panel__title">{panelTitle}</div>
            </div>
            <div className="runpage-side-panel__actions">
              <button
                type="button"
                onClick={() => setIsRunPageGuideOpen(true)}
                className="runpage-side-panel__guide"
              >
                <HelpCircle className="w-4 h-4" />
                <span>{t("runpage.guide.button") || "使用指南"}</span>
              </button>
            </div>
          </div>

          <div className="runpage-side-panel__content tab-content">
            {activeTab === "runs" && (
              <>
                {(() => {
                  const filteredRuns = runs.filter((run) => {
                    return (
                      run.status === Status.RUNNING ||
                      run.status === Status.PENDING
                    );
                  });

                  if (filteredRuns.length === 0) {
                    return (
                      <div className="flex-1 flex items-center justify-center min-h-0">
                        <Empty
                          description={
                            <div className="text-center">
                              <p className="text-muted-foreground font-medium">
                                {t("empty.no-runs")}
                              </p>
                              <p className="text-xs text-muted-foreground mt-2">
                                {t("empty.start-backend")}
                              </p>
                            </div>
                          }
                        />
                      </div>
                    );
                  }

                  return (
                    <AsTable
                      className="run-sider-table h-full w-full rounded-2xl"
                      columns={[
                        {
                          key: "id",
                          hidden: folded,
                          ellipsis: { showTitle: false },
                          render: (value, record) => (
                            <TextCell
                              text={value}
                              selected={selectedRowKeys.includes(
                                record.project,
                              )}
                            />
                          ),
                        },
                        {
                          dataIndex: "name",
                          key: "name",
                          render: (value, record) => (
                            <TextCell
                              text={value}
                              selected={selectedRowKeys.includes(
                                record.project,
                              )}
                            />
                          ),
                        },
                        {
                          key: "status",
                          render: (value, record) => (
                            <StatusCell
                              status={value}
                              selected={selectedRowKeys.includes(
                                record.project,
                              )}
                            />
                          ),
                        },
                        {
                          key: "nodeId",
                          title: t("runpage.guide.node"),
                          render: (value, record) => {
                            const nodeName =
                              nodes.find((n) => n.id === record.nodeId)?.name ||
                              record.nodeId ||
                              "-";
                            return (
                              <TextCell
                                text={nodeName}
                                selected={selectedRowKeys.includes(
                                  record.project,
                                )}
                              />
                            );
                          },
                        },
                      ]}
                      dataSource={filteredRuns}
                      rowClassName={(record) =>
                        runId === record.id ? "current-run-row" : ""
                      }
                      onRow={(record) => {
                        return {
                          "data-run-id": record.id,
                          "data-run-status": record.status,
                          onClick: (event) => {
                            if (event.type === "click") {
                              onRunClick(record.id);
                            }
                          },
                          style: {
                            cursor: "pointer",
                          },
                        };
                      }}
                      pagination={false}
                      rowKey="id"
                      rowSelection={undefined}
                      showSorterTooltip={!folded}
                      style={{
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius-xl)",
                        flex: 1,
                        overflow: "auto",
                        minHeight: 0,
                        background: "var(--card)",
                        boxShadow: "var(--shadow-sm)",
                        ...RemoveScrollBarStyle,
                      }}
                      rowHoverable={true}
                    />
                  );
                })()}
              </>
            )}

            {activeTab !== "runs" && <ResourceNodeSelector extra={renderDockerFilter()} />}
            {activeTab === "overview" && (
              <SystemOverviewSider
                data={
                  systemOverviewData
                    ? {
                        ...systemOverviewData,
                        gpuInfo: isAdmin || isAllNodes
                          ? gpuInfo || []
                          : (gpuInfo || []).filter(
                              (gpu) => gpu.nodeId === nodeId,
                            ),
                      }
                    : null
                }
                onRefreshGPUInfo={onRefreshGPUInfo}
              />
            )}

            {activeTab === "datasets" && (
              <div className="flex-1 overflow-hidden">
                <DatasetManager
                  datasets={filteredDatasets}
                  isQuerying={isQueryingDatasets}
                  hasError={datasetQueryError}
                  errorMessage={datasetErrorMessage}
                  hasQueried={hasQueriedDatasets}
                  onQuery={onQueryDatasets}
                  onRefresh={onRefreshDatasets}
                  cacheMeta={datasetCacheMeta}
                  onUpload={isAllNodes ? undefined : onUpload}
                  onDownload={onDownload}
                  onDelete={onDeleteDataset}
                  onUseForTraining={onUseDatasetForTraining}
                  onLoadPreviews={onLoadDatasetPreviews}
                  isInputDisabled={isInputDisabled}
                  inputDisabledHint={inputDisabledHint}
                  isUploading={isUploading}
                  downloadingId={downloadingId}
                  deletingId={deletingDatasetId}
                  isAllNodes={isAllNodes}
                  isMultiContainerQuery={isAdmin}
                  currentRunNodeId={currentRunNodeId}
                  currentRunContainerName={currentTrainingContainerName}
                />
              </div>
            )}

            {activeTab === "models" && (
              <div className="flex-1 overflow-hidden">
                <ModelManager
                  models={filteredModels}
                  isQuerying={isQueryingModels}
                  hasError={modelQueryError}
                  errorMessage={modelErrorMessage}
                  hasQueried={hasQueriedModels}
                  onQuery={onQueryModels}
                  onRefresh={onRefreshModels}
                  cacheMeta={modelCacheMeta}
                  onDelete={onDeleteModel}
                  deletingId={deletingModelId}
                  isAllNodes={isAllNodes}
                  isMultiContainerQuery={isAdmin}
                />
              </div>
            )}

            {activeTab === "evaluation" && (
              <div className="flex-1 overflow-hidden">
                <EvaluationManager
                  tests={filteredTests}
                  isQuerying={isQueryingTests}
                  hasError={testQueryError}
                  errorMessage={testErrorMessage}
                  hasQueried={hasQueriedTests}
                  onQuery={onQueryTests}
                  onRefresh={onRefreshTests}
                  cacheMeta={testCacheMeta}
                  onUpload={isAllNodes ? undefined : onUploadTest}
                  onDownload={onDownloadTest}
                  onDelete={onDeleteTest}
                  isUploading={isUploadingTest}
                  downloadingId={downloadingTestId}
                  deletingId={deletingTestId}
                  results={filteredEvaluationResults}
                  isQueryingResults={isQueryingEvaluationResults}
                  hasQueriedResults={hasQueriedEvaluationResults}
                  onQueryResults={onQueryEvaluationResults || (() => {})}
                  onRefreshResults={onRefreshEvaluationResults}
                  resultCacheMeta={evaluationResultCacheMeta}
                  onDownloadResult={onDownloadEvaluationResult}
                  downloadingResultId={downloadingResultId}
                  onDeleteResult={onDeleteEvaluationResult}
                  deletingResultId={deletingResultId}
                  onUseEvaluationForBenchmark={onUseEvaluationForBenchmark}
                  isInputDisabled={isInputDisabled}
                  inputDisabledHint={inputDisabledHint}
                  isAllNodes={isAllNodes}
                  isMultiContainerQuery={isAdmin}
                  currentRunNodeId={currentRunNodeId}
                  currentRunContainerName={currentEvaluationContainerName}
                />
              </div>
            )}
          </div>
        </div>

        <GuideModal
          title={t("runpage.guide.title") || "使用指南"}
          content={guideContent}
          open={isRunPageGuideOpen}
          onClose={() => setIsRunPageGuideOpen(false)}
        />
      </Flex>
    </div>
  );
};

export default memo(ProjectRunSider);


