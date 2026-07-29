import { memo, useEffect, useMemo, useState } from "react";
import {
  Button,
  Card,
  Spin,
  Modal,
  Form,
  Input,
  Dropdown,
  Tooltip,
  Popconfirm,
  Select,
} from "antd";
import { useTranslation } from "react-i18next";
import {
  Search,
  RefreshCw,
  Cpu,
  Box,
  CheckCircle2,
  XCircle,
  ChevronDown,
  Clock,
  Database,
  Trash2,
} from "lucide-react";
import { ManagementCacheMeta, ModelInfo } from "@shared/types";
import { useContainerMemory } from "@/hooks/useContainerMemory";
import { useEnvironmentConfig } from "@/hooks/useEnvironmentConfig";
import { GuideModal } from "@/components/GuideModal";
import { ManagerButton } from "@/components/buttons/ASButton";
import ManagerSectionHeader from "../ManagerSectionHeader";

const tagSelectClassName =
  "w-[84px] [&_.ant-select-selector]:!h-8 [&_.ant-select-selector]:!rounded-lg [&_.ant-select-selector]:!border-border/40 [&_.ant-select-selector]:!bg-muted/35 [&_.ant-select-selection-item]:!text-xs [&_.ant-select-selection-item]:!font-medium [&_.ant-select-selection-item]:!leading-[30px] [&_.ant-select-arrow]:!text-muted-foreground";
const dockerSelectClassName = "min-w-[128px]";
const MODEL_TAG_ORDER = ["base", "sft", "dpo", "inference"];

const MODEL_TYPE_TAG_MAP: Record<string, string> = {
  base_train: "base",
  batch_trained: "sft",
  daily_trained: "dpo",
  inference: "inference",
};

const getModelTags = (model: ModelInfo) => {
  const normalizedText =
    `${model.type || ""} ${model.name || ""} ${model.path || ""}`.toLowerCase();
  const tags = new Set<string>();

  if (model.type) {
    tags.add(MODEL_TYPE_TAG_MAP[model.type] || model.type);
  }
  if (normalizedText.includes("sft")) tags.add("sft");
  if (normalizedText.includes("dpo")) tags.add("dpo");

  return Array.from(tags);
};

interface Props {
  models: ModelInfo[];
  isQuerying: boolean;
  hasError: boolean;
  errorMessage?: string;
  hasQueried: boolean;
  onQuery: (containerName: string) => void;
  onRefresh?: (containerName: string) => void;
  cacheMeta?: ManagementCacheMeta | null;
  onDelete?: (model: ModelInfo) => Promise<void>;
  deletingId?: string | null;
  isAllNodes?: boolean;
  isMultiContainerQuery?: boolean;
}

const ModelManager = ({
  models,
  isQuerying,
  hasError,
  errorMessage,
  hasQueried,
  onQuery,
  onRefresh,
  cacheMeta,
  onDelete,
  deletingId,
  isAllNodes = false,
  isMultiContainerQuery = false,
}: Props) => {
  const { t } = useTranslation();
  const getModelTypeLabel = (type?: string) => {
    if (!type) return "";
    return t(`model.typeLabel.${type}`) || type;
  };
  const getModelTypeDescription = (type?: string) => {
    if (!type) return "";
    return t(`model.typeHelp.${type}`) || type;
  };
  const guideContent = [
    `1. ${t("model.guide.section1.title") || "查询模型"}`,
    `   - ${t("model.guide.section1.item1") || "使用后端默认 Docker 容器查询可用模型"}`,
    `   - ${t("model.guide.section1.item2") || "可查询以下四种类型的模型："}`,
    `     • ${t("model.guide.section1.item3") || "Base Train：基础训练模型"}`,
    `     • ${t("model.guide.section1.item4") || "Batch Trained：批量训练模型"}`,
    `     • ${t("model.guide.section1.item5") || "DPO：增强训练模型"}`,
    `     • ${t("model.guide.section1.item6") || "Inference：推理部署模型"}`,
    "",
    `2. ${t("model.guide.section2.title") || "合并状态说明"}`,
    `   - ${t("model.guide.section2.item1") || "Training Complete：模型已完成合并，可直接使用"}`,
    `   - ${t("model.guide.section2.item2") || "Training Incomplete：模型训练流程尚未完成，仍需继续处理"}`,
  ].join("\n");
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
  } = useContainerMemory("model");
  const containerName = defaultContainerName || rememberedContainerName;
  const hasContainerMismatch = Boolean(
    cacheMeta?.containerName && cacheMeta.containerName !== containerName,
  );
  const effectiveModels = hasContainerMismatch ? [] : models;
  const effectiveCacheMeta = hasContainerMismatch ? null : cacheMeta;
  const visibleModels = useMemo(() => {
    return effectiveModels.filter((model) => {
      if (model.size === "0 B") return false;
      if (
        selectedContainerFilter !== "all" &&
        model.containerName !== selectedContainerFilter
      ) {
        return false;
      }
      if (selectedTag === "all") return true;
      return getModelTags(model).includes(selectedTag);
    });
  }, [effectiveModels, selectedContainerFilter, selectedTag]);
  const modelTagOptions = useMemo(() => {
    const availableTags = new Set(
      effectiveModels
        .filter((model) => model.size !== "0 B")
        .flatMap((model) => getModelTags(model)),
    );
    const orderedTags = [
      ...MODEL_TAG_ORDER.filter((tag) => availableTags.has(tag)),
      ...Array.from(availableTags).filter(
        (tag) => !MODEL_TAG_ORDER.includes(tag),
      ),
    ];

    return [
      { label: t("tag.all-short"), value: "all" },
      ...orderedTags.map((tag) => ({ label: tag.toUpperCase(), value: tag })),
    ];
  }, [effectiveModels, t]);
  const availableModelTags = useMemo(
    () => new Set(modelTagOptions.map((option) => option.value)),
    [modelTagOptions],
  );
  const containerFilterOptions = useMemo(() => {
    const containerNames = Array.from(
      new Set(
        effectiveModels
          .map((model) => model.containerName?.trim())
          .filter((name): name is string => Boolean(name)),
      ),
    ).sort((a, b) => a.localeCompare(b));

    return [
      { label: t("tag.all-short") || "全部", value: "all" },
      ...containerNames.map((name) => ({ label: name, value: name })),
    ];
  }, [effectiveModels, t]);
  const availableContainerFilters = useMemo(
    () => new Set(containerFilterOptions.map((option) => option.value)),
    [containerFilterOptions],
  );
  const selectedTagLabel =
    modelTagOptions.find((option) => option.value === selectedTag)?.label ||
    selectedTag.toUpperCase();
  useEffect(() => {
    if (!availableModelTags.has(selectedTag)) {
      setSelectedTag("all");
    }
  }, [availableModelTags, selectedTag]);
  useEffect(() => {
    if (!availableContainerFilters.has(selectedContainerFilter)) {
      setSelectedContainerFilter("all");
    }
  }, [availableContainerFilters, selectedContainerFilter]);
  const [form] = Form.useForm();

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
              <Cpu className="w-16 h-16 mx-auto text-muted-foreground/50" />
            </div>
            <h3 className="text-lg font-semibold mb-3">
              {t("model.no-data-title") || "模型管理"}
            </h3>
            <p className="text-sm text-muted-foreground mb-5">
              {t("model.no-data-desc") ||
                "点击下方查询按钮，系统将使用后端默认 Docker 容器执行查询"}
            </p>

            <Button
              type="link"
              size="small"
              icon={<Cpu className="w-4 h-4" />}
              onClick={() => setIsGuideOpen(true)}
              className="mb-4"
            >
              {t("model.guide.button") || "模型管理使用说明"}
            </Button>

            <Button
              type="primary"
              size="large"
              icon={<Search className="w-4 h-4" />}
              onClick={handleOpenModal}
              className="w-full h-11 rounded-xl model-query-button"
            >
              {t("query.models") || "查询可用模型"}
            </Button>
          </div>
        </div>

        <Modal
          title={t("model.container-input-title") || "输入容器名称"}
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
              label={t("model.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message: t("model.container-required") || "请输入容器名称",
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
              {t("model.container-default-hint") ||
                `默认使用 ${containerName}，可直接确认或修改。系统将记住您最近使用的容器名称。`}
            </p>
          </Form>
        </Modal>

        <GuideModal
          title={t("model.guide.title") || "模型管理使用说明"}
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
              <Cpu className="w-16 h-16 mx-auto text-destructive/50" />
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

            <Button
              type="primary"
              icon={<RefreshCw className="w-4 h-4" />}
              onClick={handleOpenRefreshModal}
              className="w-full h-10 rounded-xl"
            >
              {t("query.retry") || "重试"}
            </Button>
          </div>
        </div>

        <Modal
          title={t("model.container-input-title") || "输入容器名称"}
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
              label={t("model.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message: t("model.container-required") || "请输入容器名称",
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
  }

  // 4. 查询成功但无数据 - 显示空状态 + 刷新按钮
  if (hasQueried && effectiveModels.length === 0) {
    return (
      <>
        <div
          className="flex-1 flex flex-col px-3 pt-1 pb-3 overflow-hidden h-full"
          style={{ minHeight: 0 }}
        >
          {/* 头部：标题和刷新按钮 */}
          <ManagerSectionHeader
            title={t("model.title") || "模型列表"}
            count={effectiveModels.length}
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
                <Cpu className="w-16 h-16 mx-auto text-muted-foreground/30" />
              </div>
              <p className="text-muted-foreground mb-4">
                {t("query.no-models") || "暂无可用模型"}
              </p>
            </div>
          </div>
        </div>

        <Modal
          title={t("model.container-input-title") || "输入容器名称"}
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
              label={t("model.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message: t("model.container-required") || "请输入容器名称",
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
  }

  // 5. 显示模型列表
  return (
    <>
      <div
        className="flex-1 flex flex-col px-3 pt-1 pb-3 overflow-hidden h-full"
        style={{ minHeight: 0 }}
      >
        {/* 头部：标题和刷新按钮 */}
        <ManagerSectionHeader
          title={t("model.title") || "模型列表"}
          count={visibleModels.length}
          cacheMeta={effectiveCacheMeta}
          actions={
            <>
              <Select
                size="small"
                value={selectedTag}
                options={modelTagOptions}
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
            </>
          }
          guideAction={
            <Button
              type="link"
              size="small"
              icon={<Cpu className="w-3 h-3" />}
              onClick={() => setIsGuideOpen(true)}
              className="text-xs"
            >
              {t("model.guide.button") || "模型管理使用说明"}
            </Button>
          }
        />

        <GuideModal
          title={t("model.guide.title") || "模型管理使用说明"}
          content={guideContent}
          open={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
        />

        {/* 模型卡片网格 - 过滤掉大小为 0 B 的模型 */}
        <div className="flex-1 overflow-y-auto pr-4 min-h-0">
          {visibleModels.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center rounded-2xl border border-border/25 bg-muted/15 px-8 py-8">
                <Cpu className="w-14 h-14 mx-auto mb-4 text-muted-foreground/30" />
                <p className="text-sm text-muted-foreground">
                  {selectedTag === "all"
                    ? t("query.no-models") || "暂无可用模型"
                    : t("query.no-tagged-models-with-tag", {
                        tag: selectedTagLabel,
                      }) || "暂无匹配该标签的模型"}
                </p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2">
              {visibleModels.map((model, index) => (
                <Card
                  key={index}
                  size="small"
                  className="model-manager-card overflow-hidden rounded-2xl border border-slate-200/75 bg-[linear-gradient(180deg,rgba(255,255,255,0.98)_0%,rgba(248,250,252,0.98)_100%)] shadow-[0_18px_40px_-34px_rgba(15,23,42,0.28)] transition-all hover:-translate-y-[1px] hover:border-slate-300/80 hover:shadow-[0_24px_48px_-34px_rgba(15,23,42,0.3)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.94)_0%,rgba(17,24,39,0.98)_100%)] dark:shadow-[0_18px_40px_-34px_rgba(2,6,23,0.82)] dark:hover:border-white/15"
                >
                  <div className="flex items-start gap-3">
                    <div className="shrink-0">
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-primary/15 bg-[radial-gradient(circle_at_30%_30%,rgba(59,130,246,0.16),rgba(59,130,246,0.08))]">
                        <Box className="w-4.5 h-4.5 text-primary" />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0 space-y-2.5">
                      <div className="min-w-0">
                        <h4 className="truncate text-[18px] font-semibold tracking-[-0.01em] text-foreground">
                          {model.name}
                        </h4>
                      </div>
                      <div className="flex items-center gap-2 flex-wrap">
                        {model.type && (
                          <Tooltip title={getModelTypeDescription(model.type)}>
                            <span className="cursor-help rounded-full border border-border/20 bg-muted/70 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.04em] text-muted-foreground">
                              {getModelTypeLabel(model.type)}
                            </span>
                          </Tooltip>
                        )}
                        {isAllNodes && model.nodeName && (
                          <span className="rounded-full border border-border/20 bg-primary/8 px-2 py-0.5 text-[11px] font-medium text-primary">
                            {model.nodeName}
                          </span>
                        )}
                        {isMultiContainerQuery && model.containerName && (
                          <span className="rounded-full border border-border/20 bg-muted/70 px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                            {model.containerName}
                          </span>
                        )}
                        {/* 显示 merged 状态 - 只有 batch_trained 和 daily_trained 需要判断 */}
                        {["batch_trained", "daily_trained"].includes(
                          model.type || "",
                        ) &&
                          "merged" in model && (
                            <span
                              className={`rounded-full px-2.5 py-1 text-[11px] font-medium flex items-center gap-1 border ${
                                model.merged
                                  ? "bg-green-50 text-green-700 border-green-200/70"
                                  : "bg-yellow-50 text-yellow-700 border-yellow-200/70"
                              }`}
                            >
                              {model.merged ? (
                                <>
                                  <CheckCircle2 className="w-3 h-3" />
                                  {t("model.merged") || "Training Complete"}
                                </>
                              ) : (
                                <>
                                  <XCircle className="w-3 h-3" />
                                  {t("model.not-merged") || "Training Incomplete"}
                                </>
                              )}
                            </span>
                          )}
                      </div>
                    </div>
                    {onDelete && model.canDelete !== false && (
                      <div className="shrink-0">
                        <Popconfirm
                          title={
                            t("model.delete.confirm-title") ||
                            "确认删除该模型？"
                          }
                          description={(
                            t("model.delete.confirm-desc") ||
                            "删除后无法恢复：{{name}}"
                          ).replace("{{name}}", model.name)}
                          okText={t("common-confirm") || "确认"}
                          cancelText={t("common-cancel") || "取消"}
                          okButtonProps={{ danger: true }}
                          onConfirm={() => onDelete(model)}
                        >
                          <Button
                            danger
                            type="text"
                            size="small"
                            icon={<Trash2 className="w-4 h-4" />}
                            loading={
                              deletingId === `${model.type}:${model.name}`
                            }
                            title={t("model.delete.button") || "删除"}
                            className="h-9 w-9 rounded-xl border border-transparent"
                          />
                        </Popconfirm>
                      </div>
                    )}
                  </div>
                  {/* 底部显示路径和统计信息 */}
                  {(model.path || model.name) && (
                    <div className="mt-3 border-t border-border/20 pt-3">
                      <p className="mb-2 break-all font-mono text-[11px] text-muted-foreground">
                        {model.path}/{model.name}
                      </p>
                      {((model.size && model.size !== "0 B") ||
                        model.createdAt) && (
                        <div className="flex items-center gap-3 flex-wrap text-xs text-muted-foreground">
                          {model.size && model.size !== "0 B" && (
                            <span className="flex items-center gap-1">
                              <Database className="w-3 h-3" />
                              {model.size}
                            </span>
                          )}
                          {model.createdAt && (
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {new Date(model.createdAt).toLocaleString()}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>

      <Modal
        title={t("model.container-input-title") || "输入容器名称"}
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
            label={t("model.container-name") || "Docker 容器名称"}
            rules={[
              {
                required: true,
                message: t("model.container-required") || "请输入容器名称",
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

export default memo(ModelManager);
