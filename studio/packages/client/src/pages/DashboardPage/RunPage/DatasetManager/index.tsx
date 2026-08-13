import { memo, useEffect, useMemo, useState } from "react";
import {
  Button,
  Card,
  Spin,
  Collapse,
  Modal,
  Form,
  Input,
  Dropdown,
  Tooltip,
  message,
  Popconfirm,
  Select,
} from "antd";
import { useTranslation } from "react-i18next";
import {
  Search,
  RefreshCw,
  Database,
  Folder,
  FileText,
  Upload,
  Download,
  ChevronDown,
  Clock,
  ArrowRight,
  Trash2,
} from "lucide-react";
import { DatasetInfo, ManagementCacheMeta } from "@shared/types";
import { useContainerMemory } from "@/hooks/useContainerMemory";
import { useEnvironmentConfig } from "@/hooks/useEnvironmentConfig";
import { GuideModal } from "@/components/GuideModal";
import { ManagerButton } from "@/components/buttons/ASButton";
import ManagerSectionHeader from "../ManagerSectionHeader";

const tagSelectClassName =
  "w-[84px] [&_.ant-select-selector]:!h-8 [&_.ant-select-selector]:!rounded-lg [&_.ant-select-selector]:!border-border/40 [&_.ant-select-selector]:!bg-muted/35 [&_.ant-select-selection-item]:!text-xs [&_.ant-select-selection-item]:!font-medium [&_.ant-select-selection-item]:!leading-[30px] [&_.ant-select-arrow]:!text-muted-foreground";
const dockerSelectClassName = "min-w-[128px]";
const DATASET_TAG_ORDER = ["raw", "sft", "dpo"];
const HIDDEN_DATASET_FILES = new Set([
  "preprocessing_audit.json",
  "preprocessing_summary.json",
  "score_audit.json",
  "score_summary.json",
]);
const isVisibleDatasetFile = (filename: string) =>
  !HIDDEN_DATASET_FILES.has(filename.toLowerCase());

interface Props {
  datasets: DatasetInfo[];
  isQuerying: boolean;
  hasError: boolean;
  errorMessage?: string;
  hasQueried: boolean;
  onQuery: (containerName: string) => void;
  onRefresh?: (containerName: string) => void;
  cacheMeta?: ManagementCacheMeta | null;
  onUpload?: () => void;
  onDownload?: (dataset: DatasetInfo) => void;
  onDelete?: (dataset: DatasetInfo) => Promise<void>;
  onUseForTraining?: (dataset: DatasetInfo) => void;
  onUseForPreprocess?: (dataset: DatasetInfo) => void;
  onLoadPreviews?: (dataset: DatasetInfo) => Promise<void>;
  isInputDisabled?: boolean;
  inputDisabledHint?: string;
  isUploading?: boolean;
  downloadingId?: string | null;
  deletingId?: string | null;
  isAllNodes?: boolean;
  isMultiContainerQuery?: boolean;
  currentRunNodeId?: string;
  currentRunContainerName?: string;
}

const DatasetManager = ({
  datasets,
  isQuerying,
  hasError,
  errorMessage,
  hasQueried,
  onQuery,
  onRefresh,
  cacheMeta,
  onUpload,
  onDownload,
  onDelete,
  onUseForTraining,
  onUseForPreprocess,
  onLoadPreviews,
  isInputDisabled = false,
  inputDisabledHint,
  isUploading,
  downloadingId,
  deletingId,
  isAllNodes = false,
  isMultiContainerQuery = false,
  currentRunNodeId,
  currentRunContainerName,
}: Props) => {
  const { t } = useTranslation();
  const normalizedCurrentRunNodeId = currentRunNodeId?.trim();
  const normalizedCurrentRunContainerName = currentRunContainerName?.trim();
  const hasCurrentRunNode =
    Boolean(normalizedCurrentRunNodeId) &&
    normalizedCurrentRunNodeId !== "all" &&
    normalizedCurrentRunNodeId !== "unknown";
  const hasCurrentRunContainer = Boolean(normalizedCurrentRunContainerName);
  const disabledInputHint =
    inputDisabledHint ||
    t("hint.no-running-session") ||
    "未找到运行中的会话，请先启动运行实例或选择可用运行实例。";
  const disabledActionClassName =
    "cursor-not-allowed text-slate-300 hover:text-slate-300 dark:text-slate-600 dark:hover:text-slate-600";
  const getDatasetTypeDescription = (type?: string) => {
    if (!type) return "";
    return t(`dataset.typeHelp.${type}`) || type;
  };
  const getDatasetFiles = (dataset: DatasetInfo) => {
    const previewMap = new Map(
      (dataset.filePreviews || [])
        .filter((item) => isVisibleDatasetFile(item.filename))
        .map((item) => [item.filename, item.preview]),
    );
    const orderedFiles =
      dataset.files && dataset.files.length > 0
        ? dataset.files.filter(isVisibleDatasetFile)
        : Array.from(previewMap.keys());

    return orderedFiles.map((filename) => ({
      filename,
      preview: previewMap.get(filename) || "",
    }));
  };

  const guideContent = [
    `1. ${t("dataset.guide.section1.title") || "查询数据集"}`,
    `   - ${t("dataset.guide.section1.item1") || "使用后端默认 Docker 容器查询所有可用数据集"}`,
    "",
    `2. ${t("dataset.guide.section2.title") || "上传数据集"}`,
    `   - ${t("dataset.guide.section2.item1") || "支持 .tar 或 .tar.gz 格式的压缩包"}`,
    `   - ${t("dataset.guide.section2.item2") || "文件大小限制：20MB"}`,
    `   - ${t("dataset.guide.section2.item3") || "自动解压到对应的数据集目录"}`,
    `   - ${t("dataset.guide.section2.item4") || "文件格式要求"}`,
    `     • ${t("dataset.guide.section2.item5") || "数据集目录中需要包含 .json 格式的数据文件"}`,
    `     • ${t("dataset.guide.section2.item6") || "系统会自动读取前3条数据作为预览"}`,
    "",
    `3. ${t("dataset.guide.section3.title") || "下载数据集"}`,
    `   - ${t("dataset.guide.section3.item1") || "点击数据集卡片右侧的下载按钮"}`,
    `   - ${t("dataset.guide.section3.item2") || "以 .tar.gz 格式打包下载"}`,
  ].join("\n");
  const [activePanels, setActivePanels] = useState<string[]>([]);
  const [previewLoadingIds, setPreviewLoadingIds] = useState<string[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [selectedTag, setSelectedTag] = useState("all");
  const [selectedContainerFilter, setSelectedContainerFilter] =
    useState("all");
  const [modalAction] = useState<"query" | "refresh">("query");
  const { defaultContainerName } = useEnvironmentConfig();
  const {
    containerName: rememberedContainerName,
    setContainerName,
    history,
  } = useContainerMemory("dataset");
  const containerName = defaultContainerName || rememberedContainerName;
  const hasContainerMismatch = Boolean(
    cacheMeta?.containerName && cacheMeta.containerName !== containerName,
  );
  const effectiveDatasets = hasContainerMismatch ? [] : datasets;
  const effectiveCacheMeta = hasContainerMismatch ? null : cacheMeta;
  const datasetTagOptions = useMemo(() => {
    const availableTags = new Set(
      effectiveDatasets
        .filter((dataset) => dataset.size !== "0 B")
        .map((dataset) => dataset.type)
        .filter(Boolean),
    );
    const orderedTags = [
      ...DATASET_TAG_ORDER.filter((tag) => availableTags.has(tag)),
      ...Array.from(availableTags).filter(
        (tag) => !DATASET_TAG_ORDER.includes(tag),
      ),
    ];

    return [
      { label: t("tag.all-short"), value: "all" },
      ...orderedTags.map((tag) => ({ label: tag.toUpperCase(), value: tag })),
    ];
  }, [effectiveDatasets, t]);
  const availableDatasetTags = useMemo(
    () => new Set(datasetTagOptions.map((option) => option.value)),
    [datasetTagOptions],
  );
  const selectedTagLabel =
    datasetTagOptions.find((option) => option.value === selectedTag)?.label ||
    selectedTag.toUpperCase();
  const containerFilterOptions = useMemo(() => {
    const containerNames = Array.from(
      new Set(
        effectiveDatasets
          .map((dataset) => dataset.containerName?.trim())
          .filter((name): name is string => Boolean(name)),
      ),
    ).sort((a, b) => a.localeCompare(b));

    return [
      { label: t("tag.all-short") || "全部", value: "all" },
      ...containerNames.map((name) => ({ label: name, value: name })),
    ];
  }, [effectiveDatasets, t]);
  const availableContainerFilters = useMemo(
    () => new Set(containerFilterOptions.map((option) => option.value)),
    [containerFilterOptions],
  );
  const visibleDatasets = useMemo(() => {
    return effectiveDatasets.filter((dataset) => {
      if (dataset.size === "0 B") return false;
      if (
        selectedContainerFilter !== "all" &&
        dataset.containerName !== selectedContainerFilter
      ) {
        return false;
      }
      if (selectedTag === "all") return true;
      return dataset.type === selectedTag;
    });
  }, [effectiveDatasets, selectedContainerFilter, selectedTag]);
  useEffect(() => {
    if (!availableDatasetTags.has(selectedTag)) {
      setSelectedTag("all");
    }
  }, [availableDatasetTags, selectedTag]);
  useEffect(() => {
    if (!availableContainerFilters.has(selectedContainerFilter)) {
      setSelectedContainerFilter("all");
    }
  }, [availableContainerFilters, selectedContainerFilter]);
  const [form] = Form.useForm();
  const handlePanelChange = (
    keys: string | string[],
    visibleDatasets: DatasetInfo[],
  ) => {
    const nextKeys = Array.isArray(keys) ? keys : [keys];
    setActivePanels(nextKeys);

    nextKeys.forEach((key) => {
      const index = Number(key.replace("dataset-", ""));
      const dataset = visibleDatasets[index];
      const previewCount = (dataset?.filePreviews || []).filter((item) =>
        isVisibleDatasetFile(item.filename),
      ).length;
      const jsonFileCount = (dataset?.files || []).filter(
        (file) =>
          isVisibleDatasetFile(file) &&
          file.endsWith(".json"),
      ).length;
      const hasPreviews = previewCount > 0 && previewCount >= jsonFileCount;

      if (
        !dataset ||
        hasPreviews ||
        !onLoadPreviews ||
        previewLoadingIds.includes(key)
      ) {
        return;
      }

      setPreviewLoadingIds((prev) =>
        prev.includes(key) ? prev : [...prev, key],
      );
      onLoadPreviews(dataset)
        .catch((error: any) => {
          message.error(
            error?.message ||
              t("dataset.preview-load-failed") ||
              "预览加载失败",
          );
        })
        .finally(() => {
          setPreviewLoadingIds((prev) => prev.filter((item) => item !== key));
        });
    });
  };

  const handleOpenModal = () => {
    setContainerName(containerName);
    onQuery(containerName);
  };

  const handleOpenRefreshModal = () => {
    setContainerName(containerName);
    if (onRefresh) {
      onRefresh(containerName);
      return;
    }
    onQuery(containerName);
  };

  const handleModalOk = () => {
    form.validateFields().then((values) => {
      setContainerName(values.containerName);
      setIsModalOpen(false);
      if (modalAction === "refresh" && onRefresh) {
        onRefresh(values.containerName);
        return;
      }
      onQuery(values.containerName);
    });
  };

  const handleModalCancel = () => {
    setIsModalOpen(false);
  };

  // 1. 从未查询过 - 显示初始查询界面
  if (!hasQueried && !isQuerying && !hasError) {
    return (
      <>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-sm w-full rounded-2xl border border-border/25 bg-muted/20 px-6 py-8">
            <div className="mb-6">
              <Database className="w-16 h-16 mx-auto text-muted-foreground/50" />
            </div>
            <h3 className="text-lg font-semibold mb-3">
              {t("dataset.no-data-title") || "数据集管理"}
            </h3>
            <p className="text-sm text-muted-foreground mb-5">
              {t("dataset.no-data-desc") ||
                "点击下方查询按钮，系统将使用后端默认 Docker 容器执行查询"}
            </p>

            <Button
              type="link"
              size="small"
              icon={<Database className="w-4 h-4" />}
              onClick={() => setIsGuideOpen(true)}
              className="mb-4"
            >
              {t("dataset.guide.button") || "数据集管理使用说明"}
            </Button>

            <Button
              type="primary"
              size="large"
              icon={<Search className="w-4 h-4" />}
              onClick={handleOpenModal}
              className="w-full h-11 rounded-xl dataset-query-button"
            >
              {t("query.datasets") || "查询可用数据集"}
            </Button>
          </div>
        </div>

        <Modal
          title={t("dataset.container-input-title") || "输入容器名称"}
          open={isModalOpen}
          onOk={handleModalOk}
          onCancel={handleModalCancel}
          okText={t("common-confirm") || "确认"}
          cancelText={t("common-cancel") || "取消"}
          className="upload-modal"
        >
          <Form form={form} layout="vertical" initialValues={{ containerName }}>
            <Form.Item
              name="containerName"
              label={t("dataset.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message: t("dataset.container-required") || "请输入容器名称",
                },
              ]}
            >
              <Input
                placeholder={containerName}
                suffix={
                  history.length > 1 ? (
                    <Dropdown
                      menu={{
                        items: history.map((name, index) => ({
                          key: name,
                          label: (
                            <div className="flex items-center gap-2">
                              {index === 0 ? (
                                <Clock className="w-3 h-3 text-primary" />
                              ) : null}
                              <span>{name}</span>
                            </div>
                          ),
                          onClick: () =>
                            form.setFieldsValue({ containerName: name }),
                        })),
                      }}
                      trigger={["click"]}
                    >
                      <Button
                        type="text"
                        size="small"
                        icon={<ChevronDown className="w-4 h-4" />}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </Dropdown>
                  ) : null
                }
              />
            </Form.Item>
            <p className="text-xs text-muted-foreground">
              {t("dataset.container-default-hint") ||
                `默认使用 ${containerName}，可直接确认或修改。系统将记住您最近使用的容器名称。`}
            </p>
          </Form>
        </Modal>

        <GuideModal
          title={t("dataset.guide.title") || "数据集管理使用说明"}
          content={guideContent}
          open={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
        />
      </>
    );
  }

  // 2. 显示查询中状态
  if (isQuerying) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center rounded-2xl border border-border/25 bg-muted/20 px-8 py-7">
          <Spin size="large" className="mb-4" />
          <p className="text-muted-foreground">
            {t("query.querying") || "查询中...请稍后"}
          </p>
          {!isMultiContainerQuery && (
            <p className="text-xs text-muted-foreground mt-2">
              {t("query.container") || "容器"}: {containerName}
            </p>
          )}
        </div>
      </div>
    );
  }

  // 3. 显示错误状态
  if (hasError) {
    return (
      <>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-sm w-full rounded-2xl border border-destructive/20 bg-destructive/5 px-6 py-8">
            <div className="mb-4">
              <Database className="w-16 h-16 mx-auto text-destructive/50" />
            </div>
            <p className="text-sm text-destructive mb-4">
              {t("query.query-failed") || "查询失败，请重试"}
            </p>
            {errorMessage && (
              <p className="text-xs text-destructive/80 mb-4 bg-destructive/10 p-2 rounded">
                {errorMessage}
              </p>
            )}
            {!isMultiContainerQuery && (
              <p className="text-xs text-muted-foreground mb-4">
                {t("query.container") || "容器"}: {containerName}
              </p>
            )}

            {onRefresh && (
              <Button
                type="primary"
                icon={<RefreshCw className="w-4 h-4" />}
                onClick={handleOpenRefreshModal}
                className="w-full h-10 rounded-xl"
              >
                {t("query.retry") || "重试"}
              </Button>
            )}
          </div>
        </div>

        <Modal
          title={t("dataset.container-input-title") || "输入容器名称"}
          open={isModalOpen}
          onOk={handleModalOk}
          onCancel={handleModalCancel}
          okText={t("common-confirm") || "确认"}
          cancelText={t("common-cancel") || "取消"}
          className="upload-modal"
        >
          <Form form={form} layout="vertical" initialValues={{ containerName }}>
            <Form.Item
              name="containerName"
              label={t("dataset.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message: t("dataset.container-required") || "请输入容器名称",
                },
              ]}
            >
              <Input
                placeholder={containerName}
                suffix={
                  history.length > 1 ? (
                    <Dropdown
                      menu={{
                        items: history.map((name, index) => ({
                          key: name,
                          label: (
                            <div className="flex items-center gap-2">
                              {index === 0 ? (
                                <Clock className="w-3 h-3 text-primary" />
                              ) : null}
                              <span>{name}</span>
                            </div>
                          ),
                          onClick: () =>
                            form.setFieldsValue({ containerName: name }),
                        })),
                      }}
                      trigger={["click"]}
                    >
                      <Button
                        type="text"
                        size="small"
                        icon={<ChevronDown className="w-4 h-4" />}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </Dropdown>
                  ) : null
                }
              />
            </Form.Item>
          </Form>
        </Modal>

        <GuideModal
          title={t("dataset.guide.title") || "数据集管理使用说明"}
          content={guideContent}
          open={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
        />
      </>
    );
  }

  // 4. 查询成功但无数据 - 显示空状态 + 刷新按钮
  if (hasQueried && effectiveDatasets.length === 0) {
    return (
      <>
        <div
          className="flex-1 flex flex-col px-3 pt-1 pb-3 overflow-hidden h-full"
          style={{ minHeight: 0 }}
        >
          {/* 头部：标题和刷新按钮 */}
          <ManagerSectionHeader
            title={t("dataset.title") || "数据集列表"}
            count={effectiveDatasets.length}
            cacheMeta={effectiveCacheMeta}
            actions={
              <Button
                type="default"
                size="small"
                icon={<RefreshCw className="w-4 h-4" />}
                onClick={handleOpenModal}
              >
                {t("query.refresh") || "刷新"}
              </Button>
            }
          />

          {/* 空状态显示 */}
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center rounded-2xl border border-border/25 bg-muted/15 px-8 py-8">
              <div className="mb-4">
                <Database className="w-16 h-16 mx-auto text-muted-foreground/30" />
              </div>
              <p className="text-muted-foreground mb-4">
                {t("query.no-datasets") || "暂无可用数据集"}
              </p>
            </div>
          </div>
        </div>

        <Modal
          title={t("dataset.container-input-title") || "输入容器名称"}
          open={isModalOpen}
          onOk={handleModalOk}
          onCancel={handleModalCancel}
          okText={t("common-confirm") || "确认"}
          cancelText={t("common-cancel") || "取消"}
          className="upload-modal"
        >
          <Form form={form} layout="vertical" initialValues={{ containerName }}>
            <Form.Item
              name="containerName"
              label={t("dataset.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message: t("dataset.container-required") || "请输入容器名称",
                },
              ]}
            >
              <Input
                placeholder={containerName}
                suffix={
                  history.length > 1 ? (
                    <Dropdown
                      menu={{
                        items: history.map((name, index) => ({
                          key: name,
                          label: (
                            <div className="flex items-center gap-2">
                              {index === 0 ? (
                                <Clock className="w-3 h-3 text-primary" />
                              ) : null}
                              <span>{name}</span>
                            </div>
                          ),
                          onClick: () =>
                            form.setFieldsValue({ containerName: name }),
                        })),
                      }}
                      trigger={["click"]}
                    >
                      <Button
                        type="text"
                        size="small"
                        icon={<ChevronDown className="w-4 h-4" />}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </Dropdown>
                  ) : null
                }
              />
            </Form.Item>
          </Form>
        </Modal>

        <GuideModal
          title={t("dataset.guide.title") || "数据集管理使用说明"}
          content={guideContent}
          open={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
        />
      </>
    );
  }

  // 5. 显示数据集列表
  return (
    <>
      <div
        className="flex-1 flex flex-col px-3 pt-1 pb-3 overflow-hidden h-full"
        style={{ minHeight: 0 }}
      >
        {/* 头部：标题和刷新按钮 */}
        <ManagerSectionHeader
          title={t("dataset.title") || "数据集列表"}
          count={visibleDatasets.length}
          cacheMeta={effectiveCacheMeta}
          actions={
            <>
              <Select
                size="small"
                value={selectedTag}
                options={datasetTagOptions}
                onChange={setSelectedTag}
                className={tagSelectClassName}
                popupMatchSelectWidth={false}
                aria-label={t("tag.filter") || "标签筛选"}
              />
              {false && isMultiContainerQuery && containerFilterOptions.length > 1 && (
                <Select
                  size="small"
                  value={selectedContainerFilter}
                  options={containerFilterOptions}
                  onChange={setSelectedContainerFilter}
                  className={dockerSelectClassName}
                  popupMatchSelectWidth={false}
                  aria-label={t("query.container") || "Docker"}
                />
              )}
              {onUpload && (
                <Tooltip title={t("dataset.upload.button") || "上传"}>
                  <span>
                    <ManagerButton
                      variant="primary"
                      size="sm"
                      icon={<Upload className="w-3.5 h-3.5" />}
                      onClick={onUpload}
                      disabled={isUploading}
                      loading={isUploading}
                      className="h-8 w-8 rounded-lg p-0"
                      aria-label={t("dataset.upload.button") || "上传"}
                    />
                  </span>
                </Tooltip>
              )}
              {onRefresh && (
                <Tooltip title={t("query.refresh") || "刷新"}>
                  <span>
                    <ManagerButton
                      variant="secondary"
                      size="sm"
                      icon={<RefreshCw className="w-3.5 h-3.5" />}
                      onClick={handleOpenRefreshModal}
                      className="h-8 w-8 rounded-lg p-0"
                      aria-label={t("query.refresh") || "刷新"}
                    />
                  </span>
                </Tooltip>
              )}
            </>
          }
          guideAction={
            <Button
              type="link"
              size="small"
              icon={<Database className="w-3 h-3" />}
              onClick={() => setIsGuideOpen(true)}
              className="text-xs"
            >
              {t("dataset.guide.button") || "数据集管理使用说明"}
            </Button>
          }
        />

        <GuideModal
          title={t("dataset.guide.title") || "数据集管理使用说明"}
          content={guideContent}
          open={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
        />

        {/* 数据集卡片网格 - 过滤掉大小为 0 B 的数据集 */}
        <div className="flex-1 overflow-y-auto pr-4 min-h-0">
          {visibleDatasets.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center rounded-2xl border border-border/25 bg-muted/15 px-8 py-8">
                <Database className="w-14 h-14 mx-auto mb-4 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">
                  {selectedTag === "all"
                    ? t("query.no-datasets") || "暂无可用数据集"
                    : t("query.no-tagged-datasets-with-tag", {
                        tag: selectedTagLabel,
                      }) || "暂无匹配该标签的数据集"}
                </p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2">
              {visibleDatasets.map((dataset, index) => {
                const datasetFiles = getDatasetFiles(dataset);
                const panelKey = `dataset-${index}`;
                const isExpanded = activePanels.includes(panelKey);
                const isPreviewLoading = previewLoadingIds.includes(panelKey);
                const isRawDataset =
                  (dataset.type || "").toLowerCase() === "raw";
                const canTrainDirectly = !isRawDataset;
                const datasetNodeId = dataset.nodeId?.trim();
                const datasetContainerName = dataset.containerName?.trim();
                const isOnCurrentRunNode =
                  hasCurrentRunNode &&
                  Boolean(datasetNodeId) &&
                  datasetNodeId === normalizedCurrentRunNodeId;
                const isInCurrentTrainingContainer =
                  hasCurrentRunContainer &&
                  Boolean(datasetContainerName) &&
                  datasetContainerName === normalizedCurrentRunContainerName;
                const canShowTrainAction =
                  canTrainDirectly &&
                  isOnCurrentRunNode &&
                  isInCurrentTrainingContainer;
                const canShowPreprocessAction =
                  isRawDataset &&
                  isOnCurrentRunNode &&
                  isInCurrentTrainingContainer;
                return (
                  <Card
                    key={index}
                    size="small"
                    className="dataset-manager-card overflow-hidden rounded-2xl border border-slate-200/75 bg-[linear-gradient(180deg,rgba(255,255,255,0.98)_0%,rgba(248,250,252,0.98)_100%)] shadow-[0_18px_40px_-34px_rgba(15,23,42,0.28)] transition-all hover:-translate-y-[1px] hover:border-slate-300/80 hover:shadow-[0_24px_48px_-34px_rgba(15,23,42,0.3)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.94)_0%,rgba(17,24,39,0.98)_100%)] dark:shadow-[0_18px_40px_-34px_rgba(2,6,23,0.82)] dark:hover:border-white/15"
                  >
                    <div className="flex items-start gap-3">
                      <div className="shrink-0">
                        <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-primary/15 bg-[radial-gradient(circle_at_30%_30%,rgba(59,130,246,0.16),rgba(59,130,246,0.08))]">
                          <Folder className="w-4.5 h-4.5 text-primary" />
                        </div>
                      </div>
                      <div className="flex-1 min-w-0 space-y-2.5">
                        <div className="min-w-0">
                          <h4 className="truncate text-[18px] font-semibold tracking-[-0.01em] text-foreground">
                            {dataset.name}
                          </h4>
                        </div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <Tooltip
                            title={getDatasetTypeDescription(dataset.type)}
                          >
                            <span className="cursor-help rounded-full border border-border/20 bg-muted/70 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.04em] text-muted-foreground">
                              {dataset.type}
                            </span>
                          </Tooltip>
                          {dataset.size && (
                            <span className="text-[12px] font-medium text-muted-foreground">
                              {dataset.size}
                            </span>
                          )}
                          {isAllNodes && dataset.nodeName && (
                            <span className="rounded-full border border-border/20 bg-primary/8 px-2 py-0.5 text-[11px] font-medium text-primary">
                              {dataset.nodeName}
                            </span>
                          )}
                          {isMultiContainerQuery && dataset.containerName && (
                            <span className="rounded-full border border-border/20 bg-muted/70 px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                              {dataset.containerName}
                            </span>
                          )}
                        </div>
                        <div className="grid gap-1 text-xs text-muted-foreground">
                          <div className="flex items-center gap-1.5">
                            <FileText className="h-3 w-3 shrink-0" />
                            <span>
                              {t("dataset.file-count", {
                                count: datasetFiles.length,
                              }) || `${datasetFiles.length} files`}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="shrink-0 flex items-center gap-1">
                        <Button
                          type="text"
                          size="small"
                          icon={<Download className="w-4 h-4" />}
                          onClick={() => onDownload?.(dataset)}
                          loading={downloadingId === dataset.name}
                          title={t("dataset.download.button") || "下载"}
                          className="h-9 w-9 rounded-xl border border-transparent text-slate-600 hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900 dark:text-slate-300 dark:hover:border-white/10 dark:hover:bg-slate-800/70 dark:hover:text-slate-100"
                        />
                        {onDelete && dataset.canDelete !== false && (
                          <Popconfirm
                            title={
                              t("dataset.delete.confirm-title") ||
                              "确认删除该数据集？"
                            }
                            description={
                              <div className="space-y-1 text-xs">
                                <div>
                                  {(
                                    t("dataset.delete.confirm-desc") ||
                                    "删除后无法恢复：{{name}}"
                                  ).replace("{{name}}", dataset.name)}
                                </div>
                                <div className="text-muted-foreground">
                                  {t("dataset.delete.single-container-only") ||
                                    "Only this dataset copy in this container will be deleted."}
                                </div>
                                <div className="font-mono text-muted-foreground">
                                  {t("dataset.delete.node-label") || "Node"}：
                                  {dataset.nodeName || dataset.nodeId || "-"}
                                </div>
                                <div className="font-mono text-muted-foreground">
                                  {t("dataset.delete.container-label") ||
                                    "Container"}：{dataset.containerName || "-"}
                                </div>
                                <div className="font-mono text-muted-foreground">
                                  {t("dataset.delete.type-label") || "Type"}：
                                  {dataset.type || "-"}
                                </div>
                                <div className="font-mono text-muted-foreground">
                                  {t("dataset.delete.name-label") || "Name"}：
                                  {dataset.name}
                                </div>
                              </div>
                            }
                            okText={t("common-confirm") || "确认"}
                            cancelText={t("common-cancel") || "取消"}
                            okButtonProps={{ danger: true }}
                            onConfirm={() => onDelete(dataset)}
                          >
                            <Button
                              danger
                              type="text"
                              size="small"
                              icon={<Trash2 className="w-4 h-4" />}
                              loading={
                                deletingId === `${dataset.type}:${dataset.name}`
                              }
                              title={t("dataset.delete.button") || "删除"}
                              className="h-9 w-9 rounded-xl border border-transparent"
                            />
                          </Popconfirm>
                        )}
                      </div>
                    </div>

                    {(canShowTrainAction ||
                      canShowPreprocessAction ||
                      isRawDataset) && (
                      <div className="mt-2 border-t border-border/20 pt-2">
                        {canShowTrainAction || canShowPreprocessAction ? (
                          <div className="flex items-center justify-end">
                            <Tooltip
                              title={
                                isInputDisabled ? disabledInputHint : undefined
                              }
                            >
                              <span>
                              <button
                                type="button"
                                disabled={isInputDisabled}
                                className={`inline-flex items-center gap-1.5 rounded-full px-0 py-0 text-sm font-medium transition-colors ${
                                  isInputDisabled
                                    ? disabledActionClassName
                                    : "cursor-pointer text-primary hover:text-primary/80"
                                }`}
                                onClick={() => {
                                  if (canShowPreprocessAction) {
                                    onUseForPreprocess?.(dataset);
                                    return;
                                  }
                                  onUseForTraining?.(dataset);
                                }}
                              >
                                <span>
                                  {canShowPreprocessAction
                                    ? t("dataset.preprocess.button") ||
                                      "Preprocess with this"
                                    : t("dataset.train.button") ||
                                      "基于该数据启动训练"}
                                </span>
                                <ArrowRight className="h-4 w-4" />
                              </button>
                              </span>
                            </Tooltip>
                          </div>
                        ) : (
                          <div className="text-xs text-slate-400 dark:text-slate-500">
                            {t("dataset.train.rawHint") ||
                              "需先完成预处理后才能开始训练"}
                          </div>
                        )}
                      </div>
                    )}

                    {/* 文件列表和示例内容（可折叠） */}
                    {datasetFiles.length > 0 && (
                      <div className="mt-2 border-t border-border/20 pt-2">
                        <Collapse
                          ghost
                          activeKey={activePanels}
                          onChange={(keys) =>
                            handlePanelChange(keys as string[], visibleDatasets)
                          }
                          className="dataset-file-collapse"
                        >
                          <Collapse.Panel
                            header={
                              <div className="flex items-center justify-between gap-3 pr-1">
                                <div className="min-w-0">
                                  <div className="flex items-center gap-2">
                                    <span className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-border/20 bg-muted/55">
                                      <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                                    </span>
                                    <span className="text-sm font-semibold text-foreground">
                                      {t("dataset.file-list") || "数据文件"}
                                    </span>
                                    <span className="rounded-full bg-primary/8 px-2 py-0.5 text-[11px] font-semibold text-primary">
                                      {datasetFiles.length}
                                    </span>
                                  </div>
                                </div>
                                <span className="shrink-0 text-xs font-medium text-primary">
                                  {isExpanded
                                    ? t("dataset.collapse") || "收起"
                                    : t("dataset.expand") || "查看文件"}
                                </span>
                              </div>
                            }
                            key={panelKey}
                          >
                            <div className="space-y-3">
                              {datasetFiles.map((file) => (
                                <div
                                  key={`${dataset.name}-${file.filename}`}
                                  className="rounded-xl border border-border/20 bg-muted/20 p-3"
                                >
                                  <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
                                    <FileText className="w-3 h-3" />
                                    <span className="font-medium text-foreground break-all">
                                      {file.filename}
                                    </span>
                                  </div>
                                  {file.preview ? (
                                    <pre className="max-h-52 overflow-auto rounded-xl border border-border/20 bg-muted/40 p-3 font-mono text-xs whitespace-pre-wrap break-all">
                                      {file.preview}
                                    </pre>
                                  ) : isPreviewLoading ? (
                                    <div className="text-xs text-muted-foreground">
                                      {t("dataset.preview-loading") ||
                                        "正在加载预览..."}
                                    </div>
                                  ) : (
                                    <div className="text-xs text-muted-foreground">
                                      {t("dataset.preview-unavailable") ||
                                        "暂无预览"}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </Collapse.Panel>
                        </Collapse>
                      </div>
                    )}

                    {/* 底部显示路径和统计信息 */}
                    {(dataset.path || dataset.name) && (
                      <div className="mt-2 border-t border-border/20 pt-2">
                        <p className="mb-2 break-all font-mono text-xs text-muted-foreground">
                          {dataset.path}/{dataset.name}
                        </p>
                        {((dataset.size && dataset.size !== "0 B") ||
                          dataset.createdAt) && (
                          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                            {dataset.size && dataset.size !== "0 B" && (
                              <span className="flex items-center gap-1">
                                <Database className="w-3 h-3" />
                                {dataset.size}
                              </span>
                            )}
                            {dataset.createdAt && (
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {new Date(dataset.createdAt).toLocaleString()}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </Card>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <Modal
        title={t("dataset.container-input-title") || "输入容器名称"}
        open={isModalOpen}
        onOk={handleModalOk}
        onCancel={handleModalCancel}
        okText={t("common-confirm") || "确认"}
        cancelText={t("common-cancel") || "取消"}
        className="upload-modal"
      >
        <Form form={form} layout="vertical" initialValues={{ containerName }}>
          <Form.Item
            name="containerName"
            label={t("dataset.container-name") || "Docker 容器名称"}
            rules={[
              {
                required: true,
                message: t("dataset.container-required") || "请输入容器名称",
              },
            ]}
          >
            <Input
              placeholder={containerName}
              suffix={
                history.length > 1 ? (
                  <Dropdown
                    menu={{
                      items: history.map((name, index) => ({
                        key: name,
                        label: (
                          <div className="flex items-center gap-2">
                            {index === 0 ? (
                              <Clock className="w-3 h-3 text-primary" />
                            ) : null}
                            <span>{name}</span>
                          </div>
                        ),
                        onClick: () =>
                          form.setFieldsValue({ containerName: name }),
                      })),
                    }}
                    trigger={["click"]}
                  >
                    <Button
                      type="text"
                      size="small"
                      icon={<ChevronDown className="w-4 h-4" />}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Dropdown>
                ) : null
              }
            />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

export default memo(DatasetManager);
