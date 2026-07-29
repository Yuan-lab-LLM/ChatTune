import {
  memo,
  useState,
  useMemo,
  useEffect,
  useRef,
  useLayoutEffect,
} from "react";
import {
  Button,
  Card,
  Spin,
  Tag,
  Empty,
  Modal,
  Form,
  Input,
  Dropdown,
  Table,
  Space,
  Select,
  Checkbox,
  Tooltip,
  Popconfirm,
} from "antd";
import { useTranslation } from "react-i18next";
import {
  Search,
  RefreshCw,
  FileText,
  Upload,
  Download,
  Beaker,
  ChevronDown,
  Clock,
  FileDown,
  Filter,
  Settings,
  X,
  Trash2,
  ArrowRight,
} from "lucide-react";
import {
  MedicalTestFile,
  EvaluationResult,
  ManagementCacheMeta,
} from "@shared/types";
import { useContainerMemory } from "@/hooks/useContainerMemory";
import { useEnvironmentConfig } from "@/hooks/useEnvironmentConfig";
import { GuideModal } from "@/components/GuideModal";
import { ManagerButton } from "@/components/buttons/ASButton";
import ManagerSectionHeader from "../ManagerSectionHeader";
import type { ColumnsType } from "antd/es/table";

interface Props {
  tests: MedicalTestFile[];
  isQuerying: boolean;
  hasError: boolean;
  errorMessage?: string;
  hasQueried: boolean;
  onQuery: (containerName: string) => void;
  onRefresh?: (containerName: string) => void;
  cacheMeta?: ManagementCacheMeta | null;
  onUpload?: () => void;
  onDownload?: (testName: string, test?: MedicalTestFile) => void;
  onDelete?: (testName: string, test?: MedicalTestFile) => Promise<void>;
  isUploading?: boolean;
  downloadingId?: string | null;
  deletingId?: string | null;
  // 评测结果相关
  results: EvaluationResult[];
  isQueryingResults: boolean;
  hasQueriedResults: boolean;
  onQueryResults: (containerName: string) => void;
  onRefreshResults?: (containerName: string) => void;
  resultCacheMeta?: ManagementCacheMeta | null;
  onDownloadResult?: (
    folderPath: string,
    filename: string,
    result?: EvaluationResult,
  ) => void;
  downloadingResultId?: string | null;
  onDeleteResult?: (
    folderPath: string,
    result?: EvaluationResult,
  ) => Promise<void>;
  deletingResultId?: string | null;
  onUseEvaluationForBenchmark?: (testName: string) => void;
  isInputDisabled?: boolean;
  inputDisabledHint?: string;
  isAllNodes?: boolean;
  isMultiContainerQuery?: boolean;
  currentRunNodeId?: string;
  currentRunContainerName?: string;
}

// 评测类型配置
const getTestTypeConfig = (
  t: (key: string) => string,
): Record<string, { label: string; color: string }> => ({
  exam2021: {
    label: t("evaluation.type.exam2021") || "中国执业医师资格考试",
    color: "blue",
  },
  exam2024: {
    label: t("evaluation.type.exam2024") || "临床医学综合能力(西医)",
    color: "green",
  },
  usmle: {
    label: t("evaluation.type.usmle") || "美国执业医师考试",
    color: "purple",
  },
  medbench: {
    label: t("evaluation.type.medbench") || "MedBench评测",
    color: "orange",
  },
  general: { label: t("evaluation.type.general") || "通用评测", color: "cyan" },
  other: { label: t("evaluation.type.other") || "其他", color: "default" },
});

type EvaluationCategoryFilter = "all" | "medical" | "general";
type TestTypeConfig = { label: string; color: string };

// 格式化文件大小（字节 -> KB/MB/GB）
const formatFileSize = (size?: string | number): string => {
  if (!size) return "";

  // 如果已经是格式化后的字符串（如 "1.5MB"），直接返回
  if (typeof size === "string" && !/^\d+$/.test(size)) {
    return size;
  }

  const bytes = typeof size === "string" ? parseInt(size, 10) : size;
  if (isNaN(bytes) || bytes === 0) return "0 B";

  const units = ["B", "KB", "MB", "GB", "TB"];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const unit = units[Math.min(i, units.length - 1)];
  const value = bytes / Math.pow(k, i);

  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`;
};

// 根据文件名推断医疗评测类型
const inferMedicalTestType = (
  filename: string,
  type?: string,
): string | null => {
  const normalizedName = filename.toLowerCase();
  const normalizedType = (type || "").toLowerCase();

  if (
    normalizedName.includes("usmle") ||
    normalizedName.includes("step") ||
    normalizedType.includes("usmle") ||
    normalizedType.includes("step")
  ) {
    return "usmle";
  }
  if (
    normalizedName.includes("exam2024") ||
    normalizedName.includes("2024") ||
    normalizedType.includes("exam2024")
  ) {
    return "exam2024";
  }
  if (
    normalizedName.includes("exam2021") ||
    normalizedName.includes("2021") ||
    normalizedType.includes("exam2021")
  ) {
    return "exam2021";
  }
  if (
    normalizedName.includes("medbench") ||
    normalizedType.includes("medbench")
  ) {
    return "medbench";
  }

  return null;
};

const getEvaluationCategory = (test: MedicalTestFile): "medical" | "general" =>
  test.category === "general" || test.type === "general"
    ? "general"
    : "medical";

const getGeneralEvaluationTypeConfig = (
  testName: string,
  t: (key: string) => string,
): TestTypeConfig => {
  const normalizedName = testName.toLowerCase();
  const categoryMap: Record<string, TestTypeConfig> = {
    mmlu: {
      label: t("evaluation.generalType.general-knowledge") || "通用知识",
      color: "cyan",
    },
    "mmlu-pro": {
      label: t("evaluation.generalType.general-knowledge") || "通用知识",
      color: "cyan",
    },
    "c-eval": {
      label: t("evaluation.generalType.chinese") || "中文专项",
      color: "geekblue",
    },
    cmmlu: {
      label: t("evaluation.generalType.chinese") || "中文专项",
      color: "geekblue",
    },
    gpqa: {
      label: t("evaluation.generalType.advanced-reasoning") || "高阶推理",
      color: "purple",
    },
    arc: {
      label: t("evaluation.generalType.advanced-reasoning") || "高阶推理",
      color: "purple",
    },
    bbh: {
      label: t("evaluation.generalType.complex-tasks") || "复杂任务",
      color: "magenta",
    },
    gsm8k: {
      label: t("evaluation.generalType.math-reasoning") || "数学推理",
      color: "blue",
    },
    math: {
      label: t("evaluation.generalType.math-reasoning") || "数学推理",
      color: "blue",
    },
    humaneval: {
      label: t("evaluation.generalType.code-generation") || "代码生成",
      color: "gold",
    },
    livecodebench: {
      label: t("evaluation.generalType.code-generation") || "代码生成",
      color: "gold",
    },
    squad: {
      label: t("evaluation.generalType.reading-comprehension") || "阅读理解",
      color: "green",
    },
    drop: {
      label: t("evaluation.generalType.reading-comprehension") || "阅读理解",
      color: "green",
    },
    ifeval: {
      label: t("evaluation.generalType.program-control") || "程序控制",
      color: "lime",
    },
    truthfulqa: {
      label: t("evaluation.generalType.safety") || "模型安全",
      color: "red",
    },
  };

  return (
    categoryMap[normalizedName] || {
      label: t("evaluation.type.general") || "通用评测",
      color: "cyan",
    }
  );
};

const getEvaluationTypeConfig = (
  test: MedicalTestFile,
  t: (key: string) => string,
): TestTypeConfig => {
  if (getEvaluationCategory(test) === "general") {
    return getGeneralEvaluationTypeConfig(test.filename || "", t);
  }

  const testTypeConfig = getTestTypeConfig(t);

  // 如果 type 有效，直接使用
  if (test.type && test.type !== "other") {
    // 先尝试直接匹配 key
    if (testTypeConfig[test.type]) {
      return testTypeConfig[test.type];
    }
    // 再尝试根据 type 内容模糊匹配（处理中文类型名等情况）
    const normalizedType = test.type.toLowerCase();
    for (const [key, config] of Object.entries(testTypeConfig)) {
      if (
        normalizedType.includes(key.toLowerCase()) ||
        config.label.toLowerCase().includes(normalizedType) ||
        normalizedType.includes(config.label.toLowerCase())
      ) {
        return config;
      }
    }
  }

  // 尝试根据文件名和类型推断
  const inferredType = inferMedicalTestType(test.filename || "", test.type);
  if (inferredType) {
    return testTypeConfig[inferredType] || testTypeConfig.other;
  }

  return testTypeConfig.other;
};

const getResultStatusConfig = (
  t: (key: string) => string,
): Record<string, { color: string; text: string }> => ({
  finished: {
    color: "success",
    text: t("evaluation.results-status.finished") || "已完成",
  },
  running: {
    color: "orange",
    text: t("evaluation.results-status.running") || "运行中",
  },
  failed: {
    color: "error",
    text: t("evaluation.results-status.failed") || "失败",
  },
  unknown: {
    color: "default",
    text: t("evaluation.results-status.unknown") || "未知",
  },
});

const EvaluationManager = ({
  tests,
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
  isUploading,
  downloadingId,
  deletingId,
  // 评测结果相关
  results,
  isQueryingResults,
  hasQueriedResults,
  onQueryResults,
  onRefreshResults,
  resultCacheMeta,
  onDownloadResult,
  downloadingResultId,
  onDeleteResult,
  deletingResultId,
  onUseEvaluationForBenchmark,
  isInputDisabled = false,
  inputDisabledHint,
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
  const { defaultEvaluateContainerName } = useEnvironmentConfig();
  const resultStatusConfig = getResultStatusConfig(t);
  const disabledInputHint =
    inputDisabledHint ||
    t("hint.no-running-session") ||
    "未找到运行中的会话，请先启动运行实例或选择可用运行实例。";
  const disabledActionClassName =
    "cursor-not-allowed text-slate-300 hover:text-slate-300 dark:text-slate-600 dark:hover:text-slate-600";
  const guideContent = [
    `1. ${t("evaluation.guide.section1.title") || "查询评测"}`,
    `  - ${t("evaluation.guide.section1.item1") || "使用后端默认 Docker 容器查询可用评测"}`,
    `  ${t("evaluation.guide.section1.item2") || "评测类型："}`,
    `    - ${t("evaluation.guide.section1.item3") || "医疗评测"}`,
    `    - ${t("evaluation.guide.section1.item4") || "通用评测"}`,
    "",
    `2. ${t("evaluation.guide.section2.title") || "上传评测"}`,
    `  - ${t("evaluation.guide.section2.item1") || "支持上传标准医疗评测数据集"}`,
    `  - ${t("evaluation.guide.section2.item2") || "支持格式：.json、.jsonl"}`,
  ].join("\n");
  const [activeTab, setActiveTab] = useState("opensource");
  const {
    containerName: openSourceContainerName,
    setContainerName: setOpenSourceContainerName,
    history: openSourceHistory,
  } = useContainerMemory("evaluation-opensource", defaultEvaluateContainerName);
  const {
    containerName: resultContainerName,
    setContainerName: setResultContainerName,
    history: resultHistory,
  } = useContainerMemory("evaluation-results", defaultEvaluateContainerName);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [modalAction] = useState<"query" | "refresh" | "queryAll">("query");
  const [form] = Form.useForm();
  const [evaluationCategoryFilter, setEvaluationCategoryFilter] =
    useState<EvaluationCategoryFilter>("all");
  const [selectedContainerFilter, setSelectedContainerFilter] =
    useState("all");

  // 评测结果过滤
  const [modelFilter, setModelFilter] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [isResultFilterOpen, setIsResultFilterOpen] = useState(false);
  const resultsTableWrapperRef = useRef<HTMLDivElement | null>(null);
  const resultsTopScrollbarRef = useRef<HTMLDivElement | null>(null);
  const resultsTopScrollbarInnerRef = useRef<HTMLDivElement | null>(null);
  const hasStoredResultColumnPrefsRef = useRef(
    localStorage.getItem("evaluation_results_visible_columns") !== null,
  );

  // 列显示/隐藏配置（从 localStorage 读取）
  const [visibleColumns, setVisibleColumns] = useState<string[]>(() => {
    const saved = localStorage.getItem("evaluation_results_visible_columns");
    const defaultColumns = [
      ...(isAllNodes ? ["node", "container"] : []),
      "jobId",
      "model",
      "dataset",
      "status",
      "accuracy",
      "avgF1",
      "totalScore",
      "startTime",
      "endTime",
      "action",
    ];
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          const hasTotalScore = parsed.includes("totalScore");
          const hasAction = parsed.includes("action");
          // 根据配置和当前范围保留来源列
          let result = parsed.filter(
            (c) => c !== "node" && c !== "container",
          );
          if (isAllNodes || parsed.includes("node")) {
            result.splice(0, 0, "node");
          }
          if (parsed.includes("container") || isAllNodes) {
            result.splice(result.includes("node") ? 1 : 0, 0, "container");
          }

          if (hasTotalScore) return result;
          if (hasAction)
            return [
              ...result.filter((c) => c !== "action"),
              "totalScore",
              "action",
            ];
          return [...result, "totalScore"];
        }
        return defaultColumns;
      } catch {
        return defaultColumns;
      }
    }
    return defaultColumns;
  });

  const currentContainerName =
    defaultEvaluateContainerName ||
    (activeTab === "results" ? resultContainerName : openSourceContainerName);
  const hasTestContainerMismatch = Boolean(
    cacheMeta?.containerName &&
    cacheMeta.containerName !== currentContainerName,
  );
  const hasResultContainerMismatch = Boolean(
    resultCacheMeta?.containerName &&
    resultCacheMeta.containerName !== currentContainerName,
  );
  const effectiveTests = hasTestContainerMismatch ? [] : tests;
  const containerFilterOptions = useMemo(() => {
    const containerNames = Array.from(
      new Set(
        effectiveTests
          .map((test) => test.containerName?.trim())
          .filter((name): name is string => Boolean(name)),
      ),
    ).sort((a, b) => a.localeCompare(b));

    return [
      { label: t("tag.all-short") || "全部", value: "all" },
      ...containerNames.map((name) => ({ label: name, value: name })),
    ];
  }, [effectiveTests, t]);
  const availableContainerFilters = useMemo(
    () => new Set(containerFilterOptions.map((option) => option.value)),
    [containerFilterOptions],
  );
  useEffect(() => {
    if (!availableContainerFilters.has(selectedContainerFilter)) {
      setSelectedContainerFilter("all");
    }
  }, [availableContainerFilters, selectedContainerFilter]);
  const filteredEvaluationTests = effectiveTests.filter((test) => {
    if (
      selectedContainerFilter !== "all" &&
      test.containerName !== selectedContainerFilter
    ) {
      return false;
    }
    return (
      evaluationCategoryFilter === "all" ||
      getEvaluationCategory(test) === evaluationCategoryFilter
    );
  });
  const effectiveResults = hasResultContainerMismatch ? [] : results;
  const uniqueResultContainers = useMemo(
    () =>
      Array.from(
        new Set(
          effectiveResults
            .map((result) => result.containerName?.trim())
            .filter((name): name is string => Boolean(name)),
        ),
      ),
    [effectiveResults],
  );
  const shouldShowResultContainerColumn = uniqueResultContainers.length > 1;
  const effectiveTestCacheMeta = hasTestContainerMismatch ? null : cacheMeta;
  const effectiveResultCacheMeta = hasResultContainerMismatch
    ? null
    : resultCacheMeta;
  const currentHistory =
    activeTab === "results" ? resultHistory : openSourceHistory;
  const shouldShowGlobalEmptyState =
    !hasQueried &&
    !hasQueriedResults &&
    !isQuerying &&
    !isQueryingResults &&
    !hasError;

  const handleOpenModal = () => {
    if (activeTab === "results") {
      setResultContainerName(currentContainerName);
      onQueryResults(currentContainerName);
      return;
    }

    setOpenSourceContainerName(currentContainerName);
    onQuery(currentContainerName);
  };

  const handleOpenRefreshModal = () => {
    if (activeTab === "results") {
      setResultContainerName(currentContainerName);
      if (onRefreshResults) {
        onRefreshResults(currentContainerName);
        return;
      }
      onQueryResults(currentContainerName);
      return;
    }

    setOpenSourceContainerName(currentContainerName);
    if (onRefresh) {
      onRefresh(currentContainerName);
      return;
    }
    onQuery(currentContainerName);
  };

  const handleModalOk = () => {
    form.validateFields().then((values) => {
      if (modalAction === "queryAll") {
        setOpenSourceContainerName(values.containerName);
        setResultContainerName(values.containerName);
        setActiveTab("opensource");
        setIsModalOpen(false);
        onQuery(values.containerName);
        onQueryResults(values.containerName);
        return;
      }

      const isResultsTab = activeTab === "results";
      if (isResultsTab) {
        setResultContainerName(values.containerName);
      } else {
        setOpenSourceContainerName(values.containerName);
      }
      setIsModalOpen(false);

      if (modalAction === "refresh") {
        if (isResultsTab) {
          onRefreshResults?.(values.containerName);
        } else {
          onRefresh?.(values.containerName);
        }
      } else if (isResultsTab) {
        onQueryResults(values.containerName);
      } else {
        onQuery(values.containerName);
      }
    });
  };

  const handleModalCancel = () => {
    setIsModalOpen(false);
  };

  // 过滤后的结果
  const filteredResults = useMemo(() => {
    return effectiveResults.filter((result) => {
      const matchModel = !modelFilter || result.model === modelFilter;
      const matchDataset = !datasetFilter || result.dataset === datasetFilter;
      const matchStatus = !statusFilter || result.status === statusFilter;
      return matchModel && matchDataset && matchStatus;
    });
  }, [effectiveResults, modelFilter, datasetFilter, statusFilter]);

  // 获取唯一的模型、数据集和状态列表用于过滤提示
  const uniqueModels = useMemo(
    () => [...new Set(effectiveResults.map((r) => r.model))],
    [effectiveResults],
  );
  const uniqueDatasets = useMemo(
    () => [...new Set(effectiveResults.map((r) => r.dataset))],
    [effectiveResults],
  );
  const uniqueStatuses = useMemo(
    () => [...new Set(effectiveResults.map((r) => r.status))],
    [effectiveResults],
  );
  useEffect(() => {
    if (
      !shouldShowResultContainerColumn ||
      hasStoredResultColumnPrefsRef.current ||
      visibleColumns.includes("container")
    ) {
      return;
    }

    setVisibleColumns((prev) => {
      if (prev.includes("container")) return prev;
      const next = prev.filter((column) => column !== "container");
      const jobIdIndex = next.indexOf("jobId");
      if (jobIdIndex === -1) return ["container", ...next];
      return [
        ...next.slice(0, jobIdIndex),
        "container",
        ...next.slice(jobIdIndex),
      ];
    });
  }, [shouldShowResultContainerColumn, visibleColumns]);
  const activeResultFilterCount = [
    modelFilter,
    datasetFilter,
    statusFilter,
  ].filter(Boolean).length;
  const filterButtonLabel =
    t("evaluation.results.filter.button") === "evaluation.results.filter.button"
      ? "筛选"
      : t("evaluation.results.filter.button");

  // 评测结果表格列定义映射（完整定义，包含排序功能）
  const allColumnsMap: Record<string, any> = {
    jobId: {
      title: t("evaluation.results-table.job-id") || "评测ID",
      dataIndex: "jobId",
      key: "jobId",
      width: 170,
      ellipsis: true,
      sorter: (a: any, b: any) => (a.jobId || "").localeCompare(b.jobId || ""),
    },
    model: {
      title: t("evaluation.results-table.model") || "模型",
      dataIndex: "model",
      key: "model",
      width: 120,
      ellipsis: true,
      sorter: (a: any, b: any) => (a.model || "").localeCompare(b.model || ""),
    },
    dataset: {
      title: t("evaluation.results-table.dataset") || "数据集",
      dataIndex: "dataset",
      key: "dataset",
      width: 130,
      ellipsis: true,
      sorter: (a: any, b: any) =>
        (a.dataset || "").localeCompare(b.dataset || ""),
    },
    node: {
      title: t("evaluation.results-table.node") || "节点",
      dataIndex: "nodeName",
      key: "node",
      width: 100,
      ellipsis: true,
      sorter: (a: any, b: any) =>
        (a.nodeName || "").localeCompare(b.nodeName || ""),
      render: (nodeName: string) => nodeName || "-",
    },
    container: {
      title: t("query.container") || "容器",
      dataIndex: "containerName",
      key: "container",
      width: 130,
      ellipsis: true,
      sorter: (a: any, b: any) =>
        (a.containerName || "").localeCompare(b.containerName || ""),
      render: (containerName: string) => containerName || "-",
    },
    status: {
      title: t("evaluation.results-table.status") || "状态",
      dataIndex: "status",
      key: "status",
      width: 82,
      sorter: (a: any, b: any) =>
        (a.status || "").localeCompare(b.status || ""),
      render: (status: string) => {
        const config = resultStatusConfig[status] || resultStatusConfig.unknown;
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    accuracy: {
      title: t("evaluation.results-table.accuracy") || "正确率",
      dataIndex: "accuracy",
      key: "accuracy",
      width: 100,
      sorter: (a: any, b: any) => (a.accuracy || 0) - (b.accuracy || 0),
      render: (accuracy: number) =>
        accuracy !== undefined ? `${(accuracy * 100).toFixed(2)}%` : "-",
    },
    avgF1: {
      title: t("evaluation.results-table.avg-f1") || "F1分数",
      dataIndex: "avgF1",
      key: "avgF1",
      width: 100,
      sorter: (a: any, b: any) => (a.avgF1 || 0) - (b.avgF1 || 0),
      render: (avgF1: number) => (avgF1 !== undefined ? avgF1.toFixed(4) : "-"),
    },
    totalScore: {
      title: t("evaluation.results-table.total-score") || "总分",
      dataIndex: "totalScore",
      key: "totalScore",
      width: 100,
      sorter: (a: any, b: any) => (a.totalScore || 0) - (b.totalScore || 0),
      render: (totalScore?: number) =>
        totalScore && totalScore !== 0 ? totalScore.toFixed(1) : "——",
    },
    startTime: {
      title: t("evaluation.results-table.start-time") || "开始时间",
      dataIndex: "startTime",
      key: "startTime",
      width: 160,
      sorter: (a: any, b: any) =>
        new Date(a.startTime || 0).getTime() -
        new Date(b.startTime || 0).getTime(),
    },
    endTime: {
      title: t("evaluation.results-table.end-time") || "完成时间",
      dataIndex: "endTime",
      key: "endTime",
      width: 160,
      sorter: (a: any, b: any) => {
        const timeA = a.endTime ? new Date(a.endTime).getTime() : 0;
        const timeB = b.endTime ? new Date(b.endTime).getTime() : 0;
        return timeA - timeB;
      },
      render: (endTime?: string) => endTime || "-",
    },
    action: {
      title: t("evaluation.results-table.action") || "操作",
      key: "action",
      width: 92,
      fixed: "right",
      render: (_: any, record: any) => (
        <Space size="small">
          <Button
            type="text"
            size="small"
            icon={<FileDown className="w-4 h-4" />}
            onClick={() =>
              onDownloadResult?.(record.folderPath, "result.json", record)
            }
            loading={downloadingResultId === `${record.folderPath}/result.json`}
            title={t("evaluation.results.action.download") || "下载结果"}
          />
          {onDeleteResult && record.canDelete !== false && (
            <Popconfirm
              title={
                t("evaluation.results.action.delete-confirm-title") ||
                "确认删除该评测结果？"
              }
              description={(
                t("evaluation.results.action.delete-confirm-desc") ||
                "删除后无法恢复：{{name}}"
              ).replace("{{name}}", record.jobId || record.folderPath)}
              okText={t("common-confirm") || "确认"}
              cancelText={t("common-cancel") || "取消"}
              okButtonProps={{ danger: true }}
              onConfirm={() => onDeleteResult(record.folderPath, record)}
            >
              <Button
                danger
                type="text"
                size="small"
                icon={<Trash2 className="w-4 h-4" />}
                loading={deletingResultId === record.folderPath}
                title={t("evaluation.results.action.delete") || "删除结果"}
              />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  };

  // 动态生成可见列
  const resultColumns = useMemo(() => {
    return visibleColumns.map((key) => allColumnsMap[key]).filter(Boolean);
  }, [visibleColumns, allColumnsMap]);

  const resultsTableScrollWidth = useMemo(() => {
    return resultColumns.reduce((total: number, column: any) => {
      const width = typeof column?.width === "number" ? column.width : 120;
      return total + width;
    }, 32);
  }, [resultColumns]);

  // 保存列配置到 localStorage
  useEffect(() => {
    localStorage.setItem(
      "evaluation_results_visible_columns",
      JSON.stringify(visibleColumns),
    );
  }, [visibleColumns]);

  useLayoutEffect(() => {
    if (activeTab !== "results" || filteredResults.length === 0) {
      return;
    }

    let cleanup: (() => void) | undefined;
    const frameId = requestAnimationFrame(() => {
      const wrapper = resultsTableWrapperRef.current;
      const topScrollbar = resultsTopScrollbarRef.current;
      const topScrollbarInner = resultsTopScrollbarInnerRef.current;
      const tableContent = wrapper?.querySelector(
        ".ant-table-content, .ant-table-body",
      ) as HTMLDivElement | null;

      if (!wrapper || !topScrollbar || !topScrollbarInner || !tableContent) {
        return;
      }

      const syncScrollbar = () => {
        topScrollbarInner.style.width = `${Math.max(tableContent.scrollWidth, resultsTableScrollWidth)}px`;
        topScrollbar.scrollLeft = tableContent.scrollLeft;
      };

      let syncingFromTop = false;
      let syncingFromTable = false;

      const handleTopScroll = () => {
        if (syncingFromTable) {
          return;
        }
        syncingFromTop = true;
        tableContent.scrollLeft = topScrollbar.scrollLeft;
        requestAnimationFrame(() => {
          syncingFromTop = false;
        });
      };

      const handleTableScroll = () => {
        if (syncingFromTop) {
          return;
        }
        syncingFromTable = true;
        topScrollbar.scrollLeft = tableContent.scrollLeft;
        requestAnimationFrame(() => {
          syncingFromTable = false;
        });
      };

      syncScrollbar();
      tableContent.addEventListener("scroll", handleTableScroll, {
        passive: true,
      });
      topScrollbar.addEventListener("scroll", handleTopScroll, {
        passive: true,
      });
      window.addEventListener("resize", syncScrollbar);

      cleanup = () => {
        tableContent.removeEventListener("scroll", handleTableScroll);
        topScrollbar.removeEventListener("scroll", handleTopScroll);
        window.removeEventListener("resize", syncScrollbar);
      };
    });

    return () => {
      cancelAnimationFrame(frameId);
      cleanup?.();
    };
  }, [
    activeTab,
    filteredResults.length,
    resultsTableScrollWidth,
    visibleColumns,
  ]);

  // 1. 评测管理全局初始空状态
  if (shouldShowGlobalEmptyState) {
    return (
      <>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-sm w-full rounded-2xl border border-border/25 bg-muted/20 px-6 py-8">
            <div className="mb-6">
              <Beaker className="w-16 h-16 mx-auto text-muted-foreground/50" />
            </div>
            <h3 className="text-lg font-semibold mb-3">
              {t("evaluation.no-data-title") || "评测管理"}
            </h3>
            <p className="text-sm text-muted-foreground mb-5">
              {t("evaluation.no-data-desc") ||
                "当前没有本地评测缓存，可通过模板发起评测，或手动查询开源评测与评测结果。"}
            </p>
            <Button
              type="link"
              size="small"
              icon={<Beaker className="w-4 h-4" />}
              onClick={() => setIsGuideOpen(true)}
              className="mb-4"
            >
              {t("evaluation.guide.button") || "评测管理使用说明"}
            </Button>
            <Button
              type="primary"
              size="large"
              icon={<RefreshCw className="w-4 h-4" />}
              onClick={() => {
                const containerName =
                  defaultEvaluateContainerName || openSourceContainerName;
                setActiveTab("opensource");
                setOpenSourceContainerName(containerName);
                setResultContainerName(containerName);
                onQuery(containerName);
                onQueryResults(containerName);
              }}
              className="w-full h-11 rounded-xl evaluation-query-button"
            >
              {t("evaluation.query-available") || "查询可用评测"}
            </Button>
          </div>
        </div>

        <Modal
          title={t("evaluation.container-input-title") || "输入容器名称"}
          open={isModalOpen}
          onOk={handleModalOk}
          onCancel={handleModalCancel}
          okText={t("common-confirm") || "确认"}
          cancelText={t("common-cancel") || "取消"}
          className="upload-modal"
        >
          <Form
            form={form}
            layout="vertical"
            initialValues={{ containerName: currentContainerName }}
          >
            <Form.Item
              name="containerName"
              label={t("evaluation.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message:
                    t("evaluation.container-required") || "请输入容器名称",
                },
              ]}
            >
              <Input
                placeholder={currentContainerName}
                suffix={
                  currentHistory.length > 1 ? (
                    <Dropdown
                      menu={{
                        items: currentHistory.map((name, index) => ({
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
              {t("evaluation.container-default-hint") ||
                `默认使用 ${currentContainerName}，可直接确认或修改。系统将记住您最近使用的容器名称。`}
            </p>
          </Form>
        </Modal>

        <GuideModal
          title={t("evaluation.guide.title") || "评测管理使用说明"}
          content={guideContent}
          open={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
        />
      </>
    );
  }

  // 2. 开源医疗评测首次查询中
  const isFirstTimeQuerying =
    activeTab === "opensource" && !hasQueried && isQuerying;
  if (isFirstTimeQuerying) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center rounded-2xl border border-border/25 bg-muted/20 px-8 py-7">
          <Spin size="large" className="mb-4" />
          <p className="text-muted-foreground">
            {t("query.querying") || "查询中...请稍后"}
          </p>
        </div>
      </div>
    );
  }

  // 3. 开源医疗评测错误状态
  if (activeTab === "opensource" && hasError) {
    return (
      <>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-sm w-full rounded-2xl border border-destructive/20 bg-destructive/5 px-6 py-8">
            <div className="mb-4">
              <Beaker className="w-16 h-16 mx-auto text-destructive/50" />
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
                {t("query.container") || "容器"}: {openSourceContainerName}
              </p>
            )}

            <Button
              type="primary"
              icon={<RefreshCw className="w-4 h-4" />}
              onClick={handleOpenModal}
              className="w-full h-10 rounded-xl"
            >
              {t("query.retry") || "重试"}
            </Button>
          </div>
        </div>

        <Modal
          title={t("evaluation.container-input-title") || "输入容器名称"}
          open={isModalOpen}
          onOk={handleModalOk}
          onCancel={handleModalCancel}
          okText={t("common-confirm") || "确认"}
          cancelText={t("common-cancel") || "取消"}
          className="upload-modal"
        >
          <Form
            form={form}
            layout="vertical"
            initialValues={{ containerName: currentContainerName }}
          >
            <Form.Item
              name="containerName"
              label={t("evaluation.container-name") || "Docker 容器名称"}
              rules={[
                {
                  required: true,
                  message:
                    t("evaluation.container-required") || "请输入容器名称",
                },
              ]}
            >
              <Input
                placeholder={currentContainerName}
                suffix={
                  currentHistory.length > 1 ? (
                    <Dropdown
                      menu={{
                        items: currentHistory.map((name, index) => ({
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

  // 4. 显示评测模块
  return (
    <div className="flex-1 flex flex-col h-full">
      {!shouldShowGlobalEmptyState && (
        <div className="p-2 shrink-0">
          <div className="p-1 bg-muted/30 rounded-2xl border border-border/25 flex items-center gap-0.5 max-w-lg mx-auto">
            {[
              {
                key: "opensource",
                label: t("evaluation.tab.opensource") || "开源评测",
              },
              {
                key: "results",
                label: t("evaluation.tab.results") || "评测结果",
              },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`
                                    flex-1 py-1.5 px-3 text-xs font-medium rounded-xl transition-all duration-200
                                    ${
                                      activeTab === key
                                        ? "bg-background text-foreground shadow-sm border border-border/25"
                                        : "text-muted-foreground hover:text-foreground hover:bg-background/50"
                                    }
                                `}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Tab 内容区域 */}
      <div className="flex-1 overflow-hidden min-h-0">
        {activeTab === "opensource" && (
          <div className="h-full flex flex-col">
            {/* 固定头部 - 操作按钮 */}
            <div className="px-3 pt-1 pb-1.5 shrink-0">
              <ManagerSectionHeader
                title={t("evaluation.tab.opensource") || "开源评测"}
                count={filteredEvaluationTests.length}
                cacheMeta={effectiveTestCacheMeta}
                actions={
                  <>
                    {onUpload && (
                      <Tooltip title={t("evaluation.upload.button") || "上传"}>
                        <span>
                          <ManagerButton
                            variant="primary"
                            size="sm"
                            icon={<Upload className="w-3.5 h-3.5" />}
                            onClick={onUpload}
                            disabled={isUploading}
                            loading={isUploading}
                            className="h-8 w-8 rounded-lg p-0"
                            aria-label={t("evaluation.upload.button") || "上传"}
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
                    icon={<Beaker className="w-3 h-3" />}
                    onClick={() => setIsGuideOpen(true)}
                    className="text-xs"
                  >
                    {t("evaluation.guide.button") || "评测管理使用说明"}
                  </Button>
                }
              />
              <div className="mt-2 flex items-center justify-end gap-2">
                <Select<EvaluationCategoryFilter>
                  size="small"
                  value={evaluationCategoryFilter}
                  onChange={(value) => setEvaluationCategoryFilter(value)}
                  className="min-w-[132px] text-xs [&_.ant-select-selection-item]:text-xs [&_.ant-select-item-option-content]:text-xs"
                  options={[
                    {
                      value: "all",
                      label: t("evaluation.filter.all") || "全部",
                    },
                    {
                      value: "medical",
                      label: t("evaluation.filter.medical") || "Medical",
                    },
                    {
                      value: "general",
                      label: t("evaluation.filter.general") || "General",
                    },
                  ]}
                />
                {false && isMultiContainerQuery && containerFilterOptions.length > 1 && (
                  <Select
                    size="small"
                    value={selectedContainerFilter}
                    options={containerFilterOptions}
                    onChange={setSelectedContainerFilter}
                    className="min-w-[128px] text-xs"
                    popupMatchSelectWidth={false}
                    aria-label={t("query.container") || "Docker"}
                  />
                )}
              </div>
            </div>

            {/* 可滚动列表区域 */}
            <div
              className="evaluation-benchmark-list flex-1 overflow-y-auto px-3 pr-5 min-h-0"
              style={{ scrollbarGutter: "stable" }}
            >
              {isQuerying && hasQueried ? (
                <div className="flex-1 flex items-center justify-center h-full">
                  <Spin size="large" />
                </div>
              ) : filteredEvaluationTests.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={t("evaluation.empty") || "暂无评测文件"}
                />
              ) : (
                <div className="grid grid-cols-1 gap-2 pb-2">
                  {filteredEvaluationTests.map((test, index) => {
                    const config = getEvaluationTypeConfig(test, t);
                    const testNodeId = test.nodeId?.trim();
                    const testContainerName = test.containerName?.trim();
                    const isOnCurrentRunNode =
                      hasCurrentRunNode &&
                      Boolean(testNodeId) &&
                      testNodeId === normalizedCurrentRunNodeId;
                    const isInCurrentEvaluationContainer =
                      hasCurrentRunContainer &&
                      Boolean(testContainerName) &&
                      testContainerName === normalizedCurrentRunContainerName;
                    const canShowBenchmarkAction =
                      isOnCurrentRunNode && isInCurrentEvaluationContainer;
                    if (!canShowBenchmarkAction) {
                    }

                    return (
                      <Card
                        key={index}
                        size="small"
                        className="evaluation-benchmark-card border-border/25 shadow-none hover:border-border/40 hover:shadow-sm transition-all rounded-2xl"
                      >
                        <div className="flex items-start gap-3">
                          <div className="shrink-0">
                            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center border border-primary/10">
                              <FileText className="w-4.5 h-4.5 text-primary" />
                            </div>
                          </div>
                          <div className="flex-1 min-w-0 space-y-2">
                            <div className="min-w-0">
                              <h4 className="font-medium text-foreground truncate">
                                {test.filename}
                              </h4>
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                              <Tag
                                className="evaluation-benchmark-tag"
                                color={config.color}
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  maxWidth: "min(216px, 100%)",
                                  whiteSpace: "normal",
                                  height: "auto",
                                  minHeight: 34,
                                  marginInlineEnd: 0,
                                  wordBreak: "normal",
                                  overflowWrap: "anywhere",
                                  lineHeight: "1.5",
                                }}
                              >
                                {config.label}
                              </Tag>
                              <span className="text-xs text-muted-foreground">
                                {formatFileSize(test.size)}
                              </span>
                              {isAllNodes && test.nodeName && (
                                <span className="rounded-full border border-border/20 bg-primary/8 px-2 py-0.5 text-[11px] font-medium text-primary">
                                  {test.nodeName}
                                </span>
                              )}
                              {isMultiContainerQuery && test.containerName && (
                                <span className="rounded-full border border-border/20 bg-muted/70 px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                                  {test.containerName}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="shrink-0 flex items-center gap-1">
                            <Button
                              type="text"
                              size="small"
                              icon={<Download className="w-4 h-4" />}
                              onClick={() =>
                                onDownload?.(test.filename || "", test)
                              }
                              loading={downloadingId === test.filename}
                              title={t("evaluation.download.button") || "下载"}
                            />
                            {onDelete && test.canDelete !== false && (
                              <Popconfirm
                                title={
                                  t("evaluation.delete.confirm-title") ||
                                  "确认删除该评测集？"
                                }
                                description={(
                                  t("evaluation.delete.confirm-desc") ||
                                  "删除后无法恢复：{{name}}"
                                ).replace("{{name}}", test.filename || "")}
                                okText={t("common-confirm") || "确认"}
                                cancelText={t("common-cancel") || "取消"}
                                okButtonProps={{ danger: true }}
                                onConfirm={() =>
                                  onDelete(test.filename || "", test)
                                }
                              >
                                <Button
                                  danger
                                  type="text"
                                  size="small"
                                  icon={<Trash2 className="w-4 h-4" />}
                                  loading={deletingId === test.filename}
                                  title={
                                    t("evaluation.delete.button") || "删除"
                                  }
                                />
                              </Popconfirm>
                            )}
                          </div>
                        </div>
                        {canShowBenchmarkAction && (
                          <div className="mt-2 border-t border-border/20 pt-2">
                            <div className="flex items-center justify-end">
                              <Tooltip
                                title={
                                  isInputDisabled ? disabledInputHint : undefined
                                }
                              >
                                <span>
                                <button
                                  type="button"
                                  disabled={
                                    isInputDisabled ||
                                    !onUseEvaluationForBenchmark
                                  }
                                  className={`inline-flex items-center gap-1.5 rounded-full px-0 py-0 text-sm font-medium transition-colors ${
                                    isInputDisabled ||
                                    !onUseEvaluationForBenchmark
                                      ? disabledActionClassName
                                      : "cursor-pointer text-primary hover:text-primary/80"
                                  }`}
                                  onClick={() => {
                                    onUseEvaluationForBenchmark?.(
                                      test.filename || "",
                                    );
                                  }}
                                >
                                  <span>
                                    {t("evaluation.benchmark.button") ||
                                      "运行推理基准测试"}
                                  </span>
                                  <ArrowRight className="h-4 w-4" />
                                </button>
                                </span>
                              </Tooltip>
                            </div>
                          </div>
                        )}
                      </Card>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "results" && (
          <div className="h-full flex flex-col">
            {/* 固定头部 - 刷新按钮和过滤器 */}
            <div className="flex flex-col px-3 pt-1 pb-1.5 shrink-0 gap-1">
              <ManagerSectionHeader
                title={t("evaluation.tab.results") || "评测结果"}
                count={filteredResults.length}
                cacheMeta={effectiveResultCacheMeta}
                actions={
                  onRefreshResults ? (
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
                  ) : null
                }
              />
              {/* 第二行：过滤器和列设置 */}
              <div className="evaluation-results-filter-panel flex flex-col gap-2 rounded-xl border border-border/20 bg-muted/10 p-2">
                <div className="flex flex-wrap items-center justify-end gap-2">
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    <Tooltip
                      title={`${filterButtonLabel}${activeResultFilterCount > 0 ? ` (${activeResultFilterCount})` : ""}`}
                    >
                      <span>
                        <ManagerButton
                          variant="secondary"
                          size="sm"
                          icon={<Filter className="w-4 h-4" />}
                          onClick={() => setIsResultFilterOpen((prev) => !prev)}
                          className="h-8 w-8 rounded-lg p-0"
                          aria-label={filterButtonLabel}
                        />
                      </span>
                    </Tooltip>
                    <Dropdown
                      menu={{
                        items: Object.keys(allColumnsMap).map((key) => ({
                          key,
                          label: (
                            <Checkbox
                              checked={visibleColumns.includes(key)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setVisibleColumns([...visibleColumns, key]);
                                } else {
                                  setVisibleColumns(
                                    visibleColumns.filter((c) => c !== key),
                                  );
                                }
                              }}
                            >
                              {allColumnsMap[key].title}
                            </Checkbox>
                          ),
                        })),
                      }}
                      placement="bottomRight"
                    >
                      <Tooltip
                        title={
                          t("evaluation.results.columns.button") || "列设置"
                        }
                      >
                        <span>
                          <ManagerButton
                            variant="secondary"
                            size="sm"
                            icon={<Settings className="w-4 h-4" />}
                            className="h-8 w-8 rounded-lg p-0"
                            aria-label={
                              t("evaluation.results.columns.button") || "列设置"
                            }
                          />
                        </span>
                      </Tooltip>
                    </Dropdown>
                    {activeResultFilterCount > 0 && (
                      <Tooltip
                        title={t("evaluation.results.filter.clear") || "清除"}
                      >
                        <Button
                          size="small"
                          type="text"
                          danger
                          icon={<X className="w-3 h-3" />}
                          onClick={() => {
                            setModelFilter("");
                            setDatasetFilter("");
                            setStatusFilter("");
                          }}
                          className="h-8 w-8 p-0"
                          aria-label={
                            t("evaluation.results.filter.clear") || "清除"
                          }
                        />
                      </Tooltip>
                    )}
                  </div>
                </div>
                {isResultFilterOpen && (
                  <div className="flex flex-wrap items-center gap-2 pt-1">
                    <Select
                      placeholder={
                        t("evaluation.results.filter.model-placeholder") ||
                        "选择模型"
                      }
                      value={modelFilter || undefined}
                      onChange={(value) => setModelFilter(value || "")}
                      className="w-full sm:w-[150px] lg:w-[170px]"
                      size="small"
                      allowClear
                      options={[
                        {
                          value: "",
                          label:
                            t("evaluation.results.filter.all-models") ||
                            "全部模型",
                        },
                        ...uniqueModels.map((model) => ({
                          value: model,
                          label: model,
                        })),
                      ]}
                    />
                    <Select
                      placeholder={
                        t("evaluation.results.filter.dataset-placeholder") ||
                        "选择数据集"
                      }
                      value={datasetFilter || undefined}
                      onChange={(value) => setDatasetFilter(value || "")}
                      className="w-full sm:w-[150px] lg:w-[170px]"
                      size="small"
                      allowClear
                      options={[
                        {
                          value: "",
                          label:
                            t("evaluation.results.filter.all-datasets") ||
                            "全部数据集",
                        },
                        ...uniqueDatasets.map((dataset) => ({
                          value: dataset,
                          label: dataset,
                        })),
                      ]}
                    />
                    <Select
                      placeholder={
                        t("evaluation.results.filter.status-placeholder") ||
                        "选择状态"
                      }
                      value={statusFilter || undefined}
                      onChange={(value) => setStatusFilter(value || "")}
                      className="w-full sm:w-[150px] lg:w-[170px]"
                      size="small"
                      allowClear
                      options={[
                        {
                          value: "",
                          label:
                            t("evaluation.results.filter.all-statuses") ||
                            "全部",
                        },
                        ...uniqueStatuses.map((status) => {
                          const config =
                            resultStatusConfig[status] ||
                            resultStatusConfig.unknown;
                          return { value: status, label: config.text };
                        }),
                      ]}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* 表格区域 */}
            <div className="flex-1 overflow-auto px-3">
              {isQueryingResults && hasQueriedResults ? (
                <div className="flex-1 flex items-center justify-center h-full">
                  <div className="text-center rounded-2xl border border-border/25 bg-muted/15 px-8 py-7">
                    <Spin size="large" />
                  </div>
                </div>
              ) : !hasQueriedResults ? (
                <div className="flex-1 flex items-center justify-center h-full">
                  <div className="text-center rounded-2xl border border-border/25 bg-muted/15 px-8 py-8">
                    <Beaker className="w-16 h-16 mx-auto text-muted-foreground/50 mb-4" />
                    <p className="text-sm text-muted-foreground mb-4">
                      {t("evaluation.results.empty.prompt") ||
                        "点击刷新按钮查询评测结果"}
                    </p>
                    <Button
                      type="primary"
                      icon={<RefreshCw className="w-4 h-4" />}
                      onClick={() => onQueryResults(resultContainerName)}
                      className="h-10 rounded-xl"
                    >
                      {t("evaluation.results.query-button") || "查询评测结果"}
                    </Button>
                  </div>
                </div>
              ) : filteredResults.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <div className="rounded-2xl border border-border/25 bg-muted/10 px-8 py-8">
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description={
                        effectiveResults.length === 0
                          ? t("evaluation.results.empty.no-results") ||
                            "暂无评测结果"
                          : t("evaluation.results.empty.no-filtered-results") ||
                            "没有符合过滤条件的结果"
                      }
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <div
                    ref={resultsTableWrapperRef}
                    className="rounded-2xl border border-border/20 bg-background overflow-hidden"
                  >
                    <Table
                      columns={resultColumns}
                      dataSource={filteredResults}
                      rowKey="jobId"
                      size="small"
                      pagination={{ pageSize: 10 }}
                      scroll={{ x: resultsTableScrollWidth }}
                      className="evaluation-results-table"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* 统计信息 */}
            {hasQueriedResults && effectiveResults.length > 0 && (
              <div className="evaluation-results-summary p-4 pt-3 shrink-0 border-t border-border/20">
                {filteredResults.length > 0 && (
                  <div
                    ref={resultsTopScrollbarRef}
                    className="evaluation-results-top-scrollbar mb-4"
                  >
                    <div
                      ref={resultsTopScrollbarInnerRef}
                      style={{ width: resultsTableScrollWidth }}
                    />
                  </div>
                )}
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {t("evaluation.results.summary.total", {
                      count: effectiveResults.length,
                    }) || `共 ${effectiveResults.length} 条结果`}
                    {filteredResults.length !== effectiveResults.length &&
                      ` (${t("evaluation.results.summary.filtered", { count: filteredResults.length }) || `过滤后 ${filteredResults.length} 条`})`}
                  </span>
                  <span>
                    {resultStatusConfig.finished.text}:{" "}
                    {results.filter((r) => r.status === "finished").length} |
                    {resultStatusConfig.running.text}:{" "}
                    {results.filter((r) => r.status === "running").length} |
                    {resultStatusConfig.failed.text}:{" "}
                    {results.filter((r) => r.status === "failed").length}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <GuideModal
        title={t("evaluation.guide.title") || "评测管理使用说明"}
        content={guideContent}
        open={isGuideOpen}
        onClose={() => setIsGuideOpen(false)}
      />

      {/* 容器输入 Modal */}
      <Modal
        title={t("evaluation.container-input-title") || "输入容器名称"}
        open={isModalOpen}
        onOk={handleModalOk}
        onCancel={handleModalCancel}
        okText={t("common-confirm") || "确认"}
        cancelText={t("common-cancel") || "取消"}
        className="upload-modal"
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{ containerName: currentContainerName }}
        >
          <Form.Item
            name="containerName"
            label={t("evaluation.container-name") || "Docker 容器名称"}
            rules={[
              {
                required: true,
                message: t("evaluation.container-required") || "请输入容器名称",
              },
            ]}
          >
            <Input
              placeholder={currentContainerName}
              suffix={
                currentHistory.length > 1 ? (
                  <Dropdown
                    menu={{
                      items: currentHistory.map((name, index) => ({
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
            {t("evaluation.container-default-hint") ||
              `默认使用 ${currentContainerName}，可直接确认或修改。系统将记住您最近使用的容器名称。`}
          </p>
        </Form>
      </Modal>
    </div>
  );
};

export default memo(EvaluationManager);
