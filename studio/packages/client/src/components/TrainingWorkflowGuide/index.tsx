import { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { 
    Modal, 
    Button, 
    Steps, 
    Card, 
    Typography, 
    Space, 
    Tag,
    Alert,
    List,
    Divider,
    Tabs,
    Progress,
    Input,
    Upload,
    Badge
} from 'antd';
import { 
    BookOpen, 
    Database, 
    Settings, 
    Play, 
    LineChart, 
    CheckCircle,
    Upload as UploadIcon,
    FileSearch,
    Cpu,
    ChevronRight,
    ChevronLeft,
    Sparkles,
    GraduationCap,
    MousePointerClick,
    FolderOpen,
    FileJson,
    Archive,
    Inbox,
    Lightbulb,
    RefreshCw,
    Filter,
    ArrowRight,
    MessageSquare,
    Wand2,
    Star,
    Zap,
    Layers,
    Rocket,
    Calendar,
    Timer,
    TrendingUp,
    Activity,
    AlertTriangle,
    BarChart3,
    Info,
    Search
} from 'lucide-react';
import { useEnvironmentConfig } from '@/hooks/useEnvironmentConfig';

const { Title, Text, Paragraph } = Typography;
const { Step: AntStep } = Steps;

interface TrainingWorkflowGuideProps {
    open: boolean;
    onClose: () => void;
    onComplete?: () => void;
}

// 模拟训练参数
const TRAINING_PARAMS = {
    model: '<模型名称>',
    learningRate: '2e-5',
    batchSize: 4,
    epochs: 3,
    warmupSteps: 100,
    saveSteps: 500,
};

export function TrainingWorkflowGuide({ open, onClose, onComplete }: TrainingWorkflowGuideProps) {
    const { t, i18n } = useTranslation();
    const { defaultContainerName } = useEnvironmentConfig();
    const isZh = i18n.language === 'zh';
    const [currentStep, setCurrentStep] = useState(0);
    const [completedSteps, setCompletedSteps] = useState<number[]>([]);

    const totalSteps = 4;

    const handleNext = useCallback(() => {
        if (currentStep < totalSteps - 1) {
            setCompletedSteps(prev => [...new Set([...prev, currentStep])]);
            setCurrentStep(prev => prev + 1);
        }
    }, [currentStep]);

    const handlePrev = useCallback(() => {
        if (currentStep > 0) {
            setCurrentStep(prev => prev - 1);
        }
    }, [currentStep]);

    const handleStepClick = useCallback((step: number) => {
        setCurrentStep(step);
    }, []);

    const handleComplete = useCallback(() => {
        setCompletedSteps(prev => [...new Set([...prev, currentStep])]);
        
        // 标记学习完成
        localStorage.setItem('training_guide_completed', 'true');

        onClose();
        onComplete?.();
        setTimeout(() => {
            setCurrentStep(0);
            setCompletedSteps([]);
        }, 300);
    }, [currentStep, onClose, onComplete]);

    // 步骤标题和内容
    const stepsContent = [
        {
            title: isZh ? '数据准备' : 'Data Preparation',
            icon: <Database className="w-5 h-5" />,
            description: isZh ? '选数据' : 'Pick data',
        },
        {
            title: isZh ? '数据预处理' : 'Data Preprocessing',
            icon: <Settings className="w-5 h-5" />,
            description: isZh ? '转格式' : 'Prepare data',
        },
        {
            title: isZh ? '启动训练' : 'Start Training',
            icon: <Play className="w-5 h-5" />,
            description: isZh ? '选模式' : 'Start run',
        },
        {
            title: isZh ? '监控训练' : 'Monitor Training',
            icon: <LineChart className="w-5 h-5" />,
            description: isZh ? '看指标' : 'Watch metrics',
        },
    ];

    // 渲染步骤内容
    const renderStepContent = () => {
        switch (currentStep) {
            case 0:
                return <DataPreparationStep isZh={isZh} defaultContainerName={defaultContainerName} />;
            case 1:
                return <DataPreprocessingStep isZh={isZh} />;
            case 2:
                return <StartTrainingStep isZh={isZh} />;
            case 3:
                return <MonitorTrainingStep isZh={isZh} />;
            default:
                return null;
        }
    };

    return (
        <Modal
            title={null}
            open={open}
            onCancel={onClose}
            footer={null}
            width={900}
            centered
            className="training-workflow-modal"
        >
            <div className="px-2 py-2">
                <div className="overflow-hidden rounded-[28px] border border-border/25 bg-background">
                    <div className="border-b border-border/20 bg-[linear-gradient(180deg,rgba(248,250,252,0.95),rgba(248,250,252,0.78))] px-4 py-3 sm:px-5">
                        <div className="flex flex-col gap-3">
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                                <div className="min-w-0">
                                    <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200/70 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
                                        <GraduationCap className="h-3.5 w-3.5" />
                                        {t('training.guide.title')}
                                    </div>
                                    <Title level={4} className="!mb-1 !mt-2 !text-[18px] !font-semibold !text-slate-900">
                                        {t('training.guide.title')}
                                    </Title>
                                    <Text className="block max-w-xl text-xs leading-5 !text-slate-500">
                                        {t('training.guide.subtitle')}
                                    </Text>
                                </div>

                                <div className="flex flex-wrap gap-2 sm:justify-end">
                                    <div className="rounded-full border border-border/20 bg-white/90 px-3 py-1 text-xs text-slate-600">
                                        {totalSteps} {isZh ? '个步骤' : 'guided steps'}
                                    </div>
                                    <div className="rounded-full border border-border/20 bg-white/90 px-3 py-1 text-xs text-slate-600">
                                        {isZh ? '适合第一次使用' : 'Good for first-time use'}
                                    </div>
                                </div>
                            </div>

                            <div className="rounded-2xl border border-border/20 bg-white/85 px-3 py-3">
                                <Steps
                                    current={currentStep}
                                    onChange={handleStepClick}
                                    className="quick-start-learning-steps"
                                    items={stepsContent.map((step, index) => ({
                                        title: (
                                            <span className={`text-sm ${currentStep === index ? 'font-semibold text-slate-900' : 'text-slate-700'}`}>
                                                {step.title}
                                            </span>
                                        ),
                                        icon: step.icon,
                                        description: (
                                            <span className="hidden text-xs text-slate-500 sm:inline">
                                                {step.description}
                                            </span>
                                        ),
                                    }))}
                                />
                            </div>
                        </div>
                    </div>

                    {/* 步骤内容区域 */}
                    <div className="min-h-[340px] bg-muted/10 p-4 sm:p-4">
                        {renderStepContent()}
                    </div>

                    {/* 底部操作按钮 */}
                    <div className="flex items-center justify-between border-t border-border/20 bg-background/90 px-4 py-3 sm:px-5">
                        <div className="flex items-center gap-2">
                            <Button
                                onClick={handlePrev}
                                disabled={currentStep === 0}
                                icon={<ChevronLeft className="w-4 h-4" />}
                                className="h-9 rounded-xl px-3"
                            >
                                {isZh ? '上一步' : 'Previous'}
                            </Button>
                        </div>

                        <div className="flex gap-2">
                            {currentStep === totalSteps - 1 ? (
                                <Button 
                                    type="primary" 
                                    onClick={handleComplete}
                                    icon={<CheckCircle className="w-4 h-4" />}
                                    className="h-9 rounded-xl px-4"
                                >
                                    {isZh ? '完成学习' : 'Complete'}
                                </Button>
                            ) : (
                                <Button 
                                    type="primary" 
                                    onClick={handleNext}
                                    icon={<ChevronRight className="w-4 h-4" />}
                                    className="h-9 rounded-xl px-4"
                                >
                                    {isZh ? '下一步' : 'Next'}
                                </Button>
                            )}
                        </div>
                    </div>
                </div>
            </div>

        </Modal>
    );
}

// 步骤 1: 数据准备
function DataPreparationStep({ isZh, defaultContainerName }: { isZh: boolean; defaultContainerName: string }) {
    return (
        <div className="space-y-6">
            <div className="rounded-3xl border border-border/20 bg-white/85 px-5 py-5">
                <div className="flex items-start gap-4">
                    <div className="rounded-2xl border border-sky-200/70 bg-sky-50 p-3">
                        <Database className="h-7 w-7 text-sky-600" />
                    </div>
                    <div className="min-w-0">
                        <Title level={4} className="!mb-2 !mt-0">
                            {isZh ? '先准备好可用数据' : 'Start with usable data'}
                        </Title>
                        <Paragraph className="!mb-0 text-muted-foreground">
                            {isZh
                                ? '这一部分会带你快速了解数据从哪里来、支持什么格式，以及第一次上传时最容易忽略的地方。'
                                : 'This section shows where datasets come from, which formats are supported, and what first-time users usually miss.'}
                        </Paragraph>
                    </div>
                </div>
            </div>

            {/* 1. 查询已有数据集 */}
            <div className="rounded-2xl border border-border/20 bg-background px-5 py-5">
                <div className="flex items-start gap-4">
                <div className="rounded-2xl border border-violet-200/70 bg-violet-50 p-3 shrink-0">
                    <FileSearch className="w-7 h-7 text-violet-600" />
                </div>
                <div className="flex-1">
                    <Title level={5} className="!mb-2">
                        {isZh ? '1. 查询已有数据集' : '1. Query Existing Datasets'}
                    </Title>
                    <Paragraph className="text-muted-foreground">
                        {isZh 
                            ? '在左侧"数据管理"标签页中，点击"查询可用数据集"按钮，输入容器名称即可查看该容器中的所有可用数据集。'
                            : 'In the "Data Management" tab on the left, click "Query Available Datasets" button and enter the container name to view all available datasets in that container.'}
                    </Paragraph>
                </div>
                </div>
            </div>

            {/* 2. 上传数据集 */}
            <div className="rounded-2xl border border-border/20 bg-background px-5 py-5">
                <div className="flex items-start gap-4">
                <div className="rounded-2xl border border-sky-200/70 bg-sky-50 p-3 shrink-0">
                    <UploadIcon className="w-7 h-7 text-sky-600" />
                </div>
                <div className="flex-1">
                    <Title level={5} className="!mb-2">
                        {isZh ? '2. 上传数据集' : '2. Upload Dataset'}
                    </Title>
                    <Paragraph className="text-muted-foreground">
                        {isZh 
                            ? '点击"上传数据集"按钮，支持 JSON 格式，数据集类型包括：'
                            : 'Click "Upload Dataset" button, supports JSON format. Dataset types include:'}
                    </Paragraph>
                    <div className="flex gap-2 mt-3">
                        <Tag className="rounded-full border-blue-200 bg-blue-50 text-blue-700">Raw</Tag>
                        <Tag className="rounded-full border-emerald-200 bg-emerald-50 text-emerald-700">SFT</Tag>
                        <Tag className="rounded-full border-amber-200 bg-amber-50 text-amber-700">DPO</Tag>
                    </div>
                </div>
                </div>
            </div>

            <Divider />

            {/* 数据格式示例 */}
            <div className="space-y-4">
                <Title level={5} className="flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-primary" />
                    {isZh ? '数据格式示例' : 'Data Format Examples'}
                </Title>

                {/* Raw 格式 */}
                <Card 
                    size="small" 
                    className="overflow-hidden rounded-2xl border border-blue-200/70 bg-blue-50/40 shadow-none"
                    title={
                        <div className="flex items-center gap-2">
                            <Tag className="rounded-full border-blue-200 bg-blue-50 text-blue-700">Raw</Tag>
                            <Text type="secondary" className="text-xs">
                                {isZh ? '原始病历数据格式' : 'Raw Medical Record Format'}
                            </Text>
                        </div>
                    }
                >
                    <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950 p-4">
                        <pre className="text-xs text-green-400 font-mono leading-relaxed">
{`{
  "主诉": "头痛1天",
  "现病史": "患者一天前开始剧烈头痛，持续1小时",
  "既往史": "否认高血压，否认心脏病",
  "个人史": "否认吸烟，否认饮酒",
  "过敏史": "无",
  "体格检查": "体温38.5",
  "辅助检查": "无",
  "检验检查": "血常规",
  "检验检查结果": "白细胞计数xxx",
  "诊断": "感冒，发烧",
  "药方处置": "药物xxx"
}`}
                        </pre>
                    </div>
                </Card>

                {/* SFT 格式 */}
                <Card 
                    size="small" 
                    className="overflow-hidden rounded-2xl border border-emerald-200/70 bg-emerald-50/40 shadow-none"
                    title={
                        <div className="flex items-center gap-2">
                            <Tag className="rounded-full border-emerald-200 bg-emerald-50 text-emerald-700">SFT</Tag>
                            <Text type="secondary" className="text-xs">
                                {isZh ? '指令微调数据格式' : 'Supervised Fine-Tuning Format'}
                            </Text>
                        </div>
                    }
                >
                    <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950 p-4">
                        <pre className="text-xs text-green-400 font-mono leading-relaxed">
{`[
  {
    "instruction": "根据患者当前的情况，给出临床诊断",
    "input": "主诉: 头痛1天,现病史: 患者一天前开始剧烈头痛,持续1小时,既往史: 否认高血压,否认心脏病,个人史: 否认吸烟,否认饮酒,过敏史: 无,体格检查: 体温38.5,辅助检查: 无,检验检查结果: 白细胞计数xxx",
    "output": "诊断: 感冒,发烧"    
  }
]`}
                        </pre>
                    </div>
                </Card>

                {/* DPO 格式 */}
                <Card 
                    size="small" 
                    className="overflow-hidden rounded-2xl border border-amber-200/70 bg-amber-50/40 shadow-none"
                    title={
                        <div className="flex items-center gap-2">
                            <Tag className="rounded-full border-amber-200 bg-amber-50 text-amber-700">DPO</Tag>
                            <Text type="secondary" className="text-xs">
                                {isZh ? '偏好对齐数据格式' : 'Direct Preference Optimization Format'}
                            </Text>
                        </div>
                    }
                >
                    <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950 p-4">
                        <pre className="text-xs text-green-400 font-mono leading-relaxed">
{`{
  "instruction": "根据病历诊断",
  "input": "主诉:解黑便11小时,现病史:昨晚20时开始无明显诱因出现解黑便2次,过敏史: 否认药物和食物过敏史,既往史:胃窦部溃疡伴出血,个人史:否认吸烟、酗酒,物理检查: 体温: 36.5℃, 脉搏: 86次/分",
  "chosen": "急性上消化道出血，高血压",
  "rejected": "贫血。"
}`}
                        </pre>
                    </div>
                </Card>
            </div>

            <Divider />

            {/* 交互式操作演示 */}
            <InteractiveUploadDemo isZh={isZh} defaultContainerName={defaultContainerName} />
        </div>
    );
}

// 交互式上传演示组件
function InteractiveUploadDemo({ isZh, defaultContainerName }: { isZh: boolean; defaultContainerName: string }) {
    const [demoStep, setDemoStep] = useState(0);
    const [datasetType, setDatasetType] = useState<'raw' | 'sft' | 'dpo' | 'pt'>('sft');

    const demoSteps = [
        { title: isZh ? '查询数据集' : 'Query Dataset', icon: <FileSearch className="w-4 h-4" /> },
        { title: isZh ? '打开上传' : 'Open Upload', icon: <FolderOpen className="w-4 h-4" /> },
        { title: isZh ? '填写表单' : 'Fill Form', icon: <MousePointerClick className="w-4 h-4" /> },
        { title: isZh ? '提交上传' : 'Submit Upload', icon: <CheckCircle className="w-4 h-4" /> },
    ];

    const handleNext = () => {
        if (demoStep < demoSteps.length - 1) {
            setDemoStep(prev => prev + 1);
        }
    };

    const handlePrev = () => {
        if (demoStep > 0) {
            setDemoStep(prev => prev - 1);
        }
    };

    const handleReset = () => {
        setDemoStep(0);
    };

    return (
        <div className="space-y-4">
            <Title level={5} className="flex items-center gap-2">
                <MousePointerClick className="w-5 h-5 text-primary" />
                {isZh ? '操作演示' : 'Operation Demo'}
            </Title>

            {/* 演示步骤导航 */}
            <div className="flex items-center gap-2 mb-4">
                {demoSteps.map((step, index) => (
                    <div key={index} className="flex items-center">
                        <Button
                            type={demoStep === index ? 'primary' : demoStep > index ? 'default' : 'dashed'}
                            size="small"
                            icon={step.icon}
                            onClick={() => setDemoStep(index)}
                            className={demoStep === index ? 'h-9 rounded-xl bg-primary px-3' : 'h-9 rounded-xl px-3'}
                        >
                            {step.title}
                        </Button>
                        {index < demoSteps.length - 1 && (
                            <ChevronRight className="w-4 h-4 mx-1 text-muted-foreground" />
                        )}
                    </div>
                ))}
            </div>

            {/* 演示内容区域 */}
            <Card 
                className="rounded-3xl border border-border/20 bg-white/85 shadow-none"
                bodyStyle={{ padding: '20px' }}
            >
                {demoStep === 0 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 1：查询已有数据集' : 'Step 1: Query Existing Datasets'}</Text>
                        </div>
                        
                        {/* 模拟侧边栏 - 只显示查询按钮 */}
                        <div className="flex gap-4">
                            <div className="w-56 flex-shrink-0 rounded-2xl border border-border/20 bg-muted/10 p-3">
                                <div className="text-xs font-medium mb-3 text-muted-foreground">{isZh ? '数据管理' : 'Data Management'}</div>
                                <Button 
                                    type="primary" 
                                    size="middle"
                                    icon={<FileSearch className="w-4 h-4 flex-shrink-0" />}
                                    className="w-full text-xs whitespace-normal h-auto py-2 leading-tight"
                                >
                                    {isZh ? '查询可用数据集' : 'Query Available Datasets'}
                                </Button>
                            </div>
                            
                            <div className="flex-1">
                                <div className="rounded-2xl border border-sky-200/70 bg-sky-50/70 p-5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="rounded-2xl border border-sky-200/70 bg-white p-2">
                                            <Lightbulb className="w-5 h-5 text-sky-600" />
                                        </div>
                                        <Text strong className="text-base text-sky-900">
                                            {isZh ? '操作指南' : 'Operation Guide'}
                                        </Text>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">1</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '进入数据管理' : 'Enter Data Management'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '点击左侧标签页' : 'Click left tab'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">2</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '点击查询按钮' : 'Click Query Button'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '查询可用数据集' : 'Query available datasets'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">3</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '输入容器名称' : 'Enter Container Name'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? `如：${defaultContainerName}` : `e.g., ${defaultContainerName}`}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">4</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '查看结果' : 'View Results'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '浏览数据集列表' : 'Browse dataset list'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 1 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 2：打开上传模态框' : 'Step 2: Open Upload Modal'}</Text>
                        </div>

                        <div className="flex gap-4">
                            <div className="w-48 rounded-2xl border border-border/20 bg-muted/10 p-3">
                                <Button 
                                    type="primary" 
                                    size="small"
                                    icon={<UploadIcon className="w-4 h-4" />}
                                >
                                    {isZh ? '上传' : 'Upload'}
                                </Button>
                            </div>

                            <div className="flex-1 flex items-center">
                                <div className="rounded-2xl border border-amber-200/70 bg-amber-50/80 p-4">
                                    <div className="flex items-start gap-3">
                                        <Lightbulb className="w-5 h-5 text-amber-500 mt-0.5" />
                                        <div>
                                            <div className="font-medium text-amber-800 dark:text-amber-200 mb-1">
                                                {isZh ? '操作提示' : 'Operation Tip'}
                                            </div>
                                            <div className="text-sm text-amber-700 dark:text-amber-300">
                                                {isZh 
                                                    ? '点击此按钮将弹出上传模态框，开始配置上传参数。'
                                                    : 'Click this button to open the upload modal and start configuring upload parameters.'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 2 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 3：填写上传表单' : 'Step 3: Fill Upload Form'}</Text>
                        </div>

                        {/* 模拟上传模态框 */}
                        <div className="mx-auto max-w-lg rounded-3xl border border-border/20 bg-background shadow-[0_18px_50px_-36px_rgba(15,23,42,0.35)]">
                            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                                <Text strong>{isZh ? '上传数据集' : 'Upload Dataset'}</Text>
                                <span className="text-muted-foreground">×</span>
                            </div>
                            
                            <div className="p-4 space-y-4">
                                {/* 容器名称 */}
                                <div>
                                    <div className="text-sm mb-1">{isZh ? 'Docker 容器名称' : 'Docker Container Name'} <span className="text-red-500">*</span></div>
                                    <Input size="small" defaultValue={defaultContainerName} disabled className="bg-gray-100" />
                                </div>

                                {/* 数据集类型 */}
                                <div>
                                    <div className="text-sm mb-1">{isZh ? '数据集类型' : 'Dataset Type'}</div>
                                    <Tabs
                                        size="small"
                                        activeKey={datasetType}
                                        onChange={(key) => setDatasetType(key as 'raw' | 'sft' | 'dpo' | 'pt')}
                                        items={[
                                            { key: 'raw', label: isZh ? '原始数据' : 'Raw' },
                                            { key: 'sft', label: 'SFT' },
                                            { key: 'dpo', label: 'DPO' },
                                            { key: 'pt', label: isZh ? '预训练文本' : 'PT Text' },
                                        ]}
                                    />
                                </div>

                                {/* 数据集名称 */}
                                <div>
                                    <div className="text-sm mb-1">{isZh ? '数据集名称' : 'Dataset Name'} <span className="text-red-500">*</span></div>
                                    <Input size="small" placeholder="20240325" />
                                    <div className="text-xs text-amber-600 mt-1 flex items-center gap-1">
                                        <Lightbulb className="w-3 h-3" />
                                        {isZh ? '建议使用 YYYYMMDD 日期格式命名，如 20240325' : 'Use YYYYMMDD date format, e.g., 20240325'}
                                    </div>
                                </div>

                                {/* 文件上传 */}
                                <div>
                                    <div className="text-sm mb-1">{isZh ? '数据文件' : 'Data File'} <span className="text-red-500">*</span></div>
                                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center bg-gray-50">
                                        <Inbox className="w-8 h-8 mx-auto text-gray-400 mb-2" />
                                        <div className="text-sm text-gray-600">{isZh ? '点击或拖拽文件到此区域上传' : 'Click or drag files here'}</div>
                                        <div className="text-xs text-gray-400 mt-1">{isZh ? '支持 .tar, .tar.gz 格式' : 'Supports .tar, .tar.gz'}</div>
                                    </div>
                                </div>

                                {/* 文件格式要求 */}
                                <div className="bg-blue-50 p-3 rounded text-xs">
                                    <div className="font-medium text-blue-800 mb-2">{isZh ? '上传要求：' : 'Requirements:'}</div>
                                    <ul className="list-disc list-inside text-blue-700 space-y-1">
                                        <li>{isZh ? '上传的文件为压缩格式（.tar 或 .tar.gz）' : 'Upload compressed files (.tar or .tar.gz)'}</li>
                                        <li>{isZh ? '解压后只能包含 .json 文件' : 'Only .json files after extraction'}</li>
                                        <li>{isZh ? '需要 dataset_info.json 文件' : 'Requires dataset_info.json file'}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div className="rounded-2xl border border-violet-200/70 bg-violet-50/70 p-5">
                            <div className="flex items-center gap-2 mb-4">
                                <div className="rounded-2xl border border-violet-200/70 bg-white p-2">
                                    <Sparkles className="w-5 h-5 text-violet-600" />
                                </div>
                                <Text strong className="text-base text-violet-900">
                                    {isZh ? '填写指南' : 'Fill Guide'}
                                </Text>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-3">
                                <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                    <Database className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
                                    <div>
                                        <div className="font-medium text-sm">{isZh ? '容器名称' : 'Container Name'}</div>
                                        <div className="text-xs text-muted-foreground mt-1">{isZh ? `默认：${defaultContainerName}` : `Default: ${defaultContainerName}`}</div>
                                    </div>
                                </div>
                                
                                <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                    <Layers className="w-5 h-5 text-green-500 mt-0.5 shrink-0" />
                                    <div>
                                        <div className="font-medium text-sm">{isZh ? '数据集类型' : 'Dataset Type'}</div>
                                        <div className="text-xs text-muted-foreground mt-1">{isZh ? 'Raw / SFT / DPO' : 'Select based on format'}</div>
                                    </div>
                                </div>
                                
                                <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                    <FileJson className="w-5 h-5 text-purple-500 mt-0.5 shrink-0" />
                                    <div>
                                        <div className="font-medium text-sm">{isZh ? '数据集名称' : 'Dataset Name'}</div>
                                        <div className="text-xs text-muted-foreground mt-1">{isZh ? 'YYYYMMDD 日期格式' : 'YYYYMMDD date format'}</div>
                                    </div>
                                </div>
                                
                                <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                    <Archive className="w-5 h-5 text-orange-500 mt-0.5 shrink-0" />
                                    <div>
                                        <div className="font-medium text-sm">{isZh ? '数据文件' : 'Data File'}</div>
                                        <div className="text-xs text-muted-foreground mt-1">{isZh ? 'tar / tar.gz 格式' : 'tar / tar.gz format'}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 3 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 4：提交上传' : 'Step 4: Submit Upload'}</Text>
                        </div>

                        {/* 美化的操作说明卡片 */}
                        <div className="rounded-3xl border border-sky-200/70 bg-sky-50/65 p-6 shadow-none">
                            {/* 标题 */}
                            <div className="flex items-center gap-3 mb-6">
                                <div className="rounded-2xl border border-sky-200/70 bg-white p-2">
                                    <UploadIcon className="w-6 h-6 text-sky-600" />
                                </div>
                                <Text strong className="text-lg text-sky-900">
                                    {isZh ? '操作说明' : 'Operation Instructions'}
                                </Text>
                            </div>

                            {/* 5步流程 */}
                            <div className="flex items-start justify-between gap-2">
                                {/* 步骤 1 */}
                                <div className="flex-1 flex flex-col items-center text-center">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 text-white flex items-center justify-center text-sm font-bold shadow-md mb-2">
                                        1
                                    </div>
                                    <div className="text-sm font-medium text-blue-800 dark:text-blue-200">
                                        {isZh ? '点击' : 'Click'}
                                    </div>
                                    <div className="text-xs text-blue-600 dark:text-blue-400 mt-0.5">
                                        {isZh ? '上传按钮' : 'Upload'}
                                    </div>
                                </div>

                                {/* 连接线 */}
                                <div className="flex items-center pt-4">
                                    <div className="w-8 h-0.5 bg-gradient-to-r from-blue-300 to-indigo-300"></div>
                                    <ChevronRight className="w-4 h-4 text-indigo-400 -ml-1" />
                                </div>

                                {/* 步骤 2 */}
                                <div className="flex-1 flex flex-col items-center text-center">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-400 to-indigo-600 text-white flex items-center justify-center text-sm font-bold shadow-md mb-2">
                                        2
                                    </div>
                                    <div className="text-sm font-medium text-indigo-800 dark:text-indigo-200">
                                        {isZh ? '系统' : 'System'}
                                    </div>
                                    <div className="text-xs text-indigo-600 dark:text-indigo-400 mt-0.5">
                                        {isZh ? '自动解压' : 'Auto Extract'}
                                    </div>
                                </div>

                                {/* 连接线 */}
                                <div className="flex items-center pt-4">
                                    <div className="w-8 h-0.5 bg-gradient-to-r from-indigo-300 to-purple-300"></div>
                                    <ChevronRight className="w-4 h-4 text-purple-400 -ml-1" />
                                </div>

                                {/* 步骤 3 */}
                                <div className="flex-1 flex flex-col items-center text-center">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-400 to-purple-600 text-white flex items-center justify-center text-sm font-bold shadow-md mb-2">
                                        3
                                    </div>
                                    <div className="text-sm font-medium text-purple-800 dark:text-purple-200">
                                        {isZh ? '列表' : 'List'}
                                    </div>
                                    <div className="text-xs text-purple-600 dark:text-purple-400 mt-0.5">
                                        {isZh ? '自动刷新' : 'Auto Refresh'}
                                    </div>
                                </div>

                                {/* 连接线 */}
                                <div className="flex items-center pt-4">
                                    <div className="w-8 h-0.5 bg-gradient-to-r from-purple-300 to-pink-300"></div>
                                    <ChevronRight className="w-4 h-4 text-pink-400 -ml-1" />
                                </div>

                                {/* 步骤 4 */}
                                <div className="flex-1 flex flex-col items-center text-center">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-400 to-pink-600 text-white flex items-center justify-center text-sm font-bold shadow-md mb-2">
                                        4
                                    </div>
                                    <div className="text-sm font-medium text-pink-800 dark:text-pink-200">
                                        {isZh ? '查看' : 'View'}
                                    </div>
                                    <div className="text-xs text-pink-600 dark:text-pink-400 mt-0.5">
                                        {isZh ? '数据集' : 'Dataset'}
                                    </div>
                                </div>

                                {/* 连接线 */}
                                <div className="flex items-center pt-4">
                                    <div className="w-8 h-0.5 bg-gradient-to-r from-pink-300 to-emerald-300"></div>
                                    <ChevronRight className="w-4 h-4 text-emerald-400 -ml-1" />
                                </div>

                                {/* 步骤 5 */}
                                <div className="flex-1 flex flex-col items-center text-center">
                                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 text-white flex items-center justify-center text-sm font-bold shadow-md mb-2">
                                        5
                                    </div>
                                    <div className="text-sm font-medium text-emerald-800 dark:text-emerald-200">
                                        {isZh ? '开始' : 'Start'}
                                    </div>
                                    <div className="text-xs text-emerald-600 dark:text-emerald-400 mt-0.5">
                                        {isZh ? '使用' : 'Use'}
                                    </div>
                                </div>
                            </div>

                        
                        </div>
                    </div>
                )}
            </Card>

            {/* 演示控制按钮 */}
            <div className="flex justify-between items-center">
                <Button 
                    size="small" 
                    onClick={handlePrev}
                    disabled={demoStep === 0}
                    icon={<ChevronLeft className="w-4 h-4" />}
                >
                    {isZh ? '上一步' : 'Previous'}
                </Button>

                <div className="flex gap-2">
                    {demoStep === demoSteps.length - 1 ? (
                        <Button size="small" onClick={handleReset} icon={<Sparkles className="w-4 h-4" />}>
                            {isZh ? '重新演示' : 'Restart'}
                        </Button>
                    ) : (
                        <Button 
                            type="primary" 
                            size="small" 
                            onClick={handleNext}
                            icon={<ChevronRight className="w-4 h-4" />}
                        >
                            {isZh ? '下一步' : 'Next'}
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}

// 步骤 2: 数据预处理
function DataPreprocessingStep({ isZh }: { isZh: boolean }) {
    return (
        <div className="space-y-6">
            {/* 页面标题 */}
            <div className="rounded-3xl border border-border/20 bg-white/85 px-5 py-5">
                <div className="flex items-start gap-4">
                <div className="rounded-2xl border border-violet-200/70 bg-violet-50 p-3 shrink-0">
                    <Settings className="w-7 h-7 text-violet-600" />
                </div>
                <div className="flex-1">
                    <Title level={5} className="!mb-2">
                        {isZh ? '数据预处理与筛选' : 'Data Preprocessing & Filtering'}
                    </Title>
                    <Paragraph className="text-muted-foreground">
                        {isZh 
                            ? '这里主要做两件事：把 Raw 数据转成训练格式，或先筛掉质量不高的数据。'
                            : 'Two things happen here: convert Raw data into training format, or filter out lower-quality samples first.'}
                    </Paragraph>
                </div>
                </div>
            </div>

            {/* 两个功能卡片 */}
            <div className="grid grid-cols-1 gap-6">
                {/* 功能 1: 数据预处理 */}
                <Card 
                    className="overflow-hidden rounded-3xl border border-sky-200/70 bg-sky-50/55 shadow-none"
                    bodyStyle={{ padding: '24px' }}
                >
                    {/* 卡片头部 */}
                    <div className="flex items-start gap-4 mb-6">
                        <div className="rounded-2xl border border-sky-200/70 bg-white p-3 shrink-0">
                            <RefreshCw className="w-7 h-7 text-sky-600" />
                        </div>
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                                <Tag className="m-0 rounded-full border-sky-200 bg-sky-50 text-sky-700">{isZh ? '格式转换' : 'Format Conversion'}</Tag>
                                <Tag className="m-0 rounded-full border-border/30 bg-white/80 text-slate-600">{isZh ? 'Raw → SFT/DPO' : 'Raw → SFT/DPO'}</Tag>
                            </div>
                            <Title level={4} className="!mb-1 !mt-0">
                                {isZh ? '数据预处理' : 'Data Preprocessing'}
                            </Title>
                            <Paragraph className="text-muted-foreground text-sm !mb-0">
                                {isZh 
                                    ? '把原始病历数据整理成 SFT 或 DPO 训练集。'
                                    : 'Turn raw medical records into SFT or DPO training data.'}
                            </Paragraph>
                        </div>
                    </div>

                    {/* 适用场景 */}
                    <div className="mb-4 rounded-2xl border border-white/70 bg-white/80 p-4">
                        <div className="flex items-center gap-2 mb-3">
                            <Wand2 className="w-4 h-4 text-blue-500" />
                            <Text strong className="text-blue-800 dark:text-blue-200">
                                {isZh ? '适用场景' : 'Use Case'}
                            </Text>
                        </div>
                        <div className="flex items-center justify-center gap-4">
                            {/* Raw 数据源 */}
                            <div className="flex items-center gap-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                                <Database className="w-4 h-4 text-blue-600" />
                                <span className="text-sm text-blue-700 dark:text-blue-300">Raw</span>
                            </div>
                            
                            {/* 分叉箭头 */}
                            <div className="flex flex-col items-center">
                                <ArrowRight className="w-5 h-5 text-blue-400 rotate-45" />
                                <ArrowRight className="w-5 h-5 text-blue-400 -rotate-45 -mt-2" />
                            </div>
                            
                            {/* 两个目标格式 */}
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/50 rounded-lg">
                                    <FileJson className="w-4 h-4 text-green-600" />
                                    <span className="text-sm text-green-700 dark:text-green-300">SFT</span>
                                </div>
                                <div className="flex items-center gap-2 px-4 py-2 bg-orange-100 dark:bg-orange-900/50 rounded-lg">
                                    <FileJson className="w-4 h-4 text-orange-600" />
                                    <span className="text-sm font-medium text-orange-700 dark:text-orange-300">DPO</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 使用方法 */}
                    <div className="mb-4 rounded-2xl border border-slate-800 bg-slate-950 p-4">
                        <div className="flex items-center gap-2 mb-3">
                            <MessageSquare className="w-4 h-4 text-green-400" />
                            <span className="text-sm font-medium text-green-400">{isZh ? '对话指令' : 'Chat Command'}</span>
                        </div>
                        <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-3">
                            <code className="block min-h-[44px] text-sm leading-6 text-green-300 font-mono">
                                执行数据预处理操作，数据类型为sft，数据格式为diagnosis
                            </code>
                        </div>
                        <div className="mt-3 text-xs text-slate-400 space-y-1">
                            <p>• {isZh ? '系统会优先识别最新的 Raw 数据目录' : 'The system prefers the latest Raw data folder'}</p>
                            <p>• {isZh ? '补充数据类型和目标能力即可开始转换' : 'Add the target type and capability, then start the conversion'}</p>
                            <p>• {isZh ? '处理完成后会生成可直接训练的数据集' : 'A training-ready dataset is generated automatically'}</p>
                        </div>
                    </div>
                </Card>

                {/* 功能 2: 数据高级筛选 */}
                <Card 
                    className="overflow-hidden rounded-3xl border border-violet-200/70 bg-violet-50/55 shadow-none"
                    bodyStyle={{ padding: '24px' }}
                >
                    {/* 卡片头部 */}
                    <div className="flex items-start gap-4 mb-6">
                        <div className="rounded-2xl border border-violet-200/70 bg-white p-3 shrink-0">
                            <Filter className="w-7 h-7 text-violet-600" />
                        </div>
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                                <Tag className="m-0 rounded-full border-violet-200 bg-violet-50 text-violet-700">{isZh ? '质量筛选' : 'Quality Filter'}</Tag>
                                <Tag className="m-0 rounded-full border-border/30 bg-white/80 text-slate-600">{isZh ? '模型打分' : 'Model Scoring'}</Tag>
                            </div>
                            <Title level={4} className="!mb-1 !mt-0">
                                {isZh ? '数据高级筛选处理' : 'Advanced Data Filtering'}
                            </Title>
                            <Paragraph className="text-muted-foreground text-sm !mb-0">
                                {isZh 
                                    ? '先给数据打分，再筛掉质量不稳定的样本。'
                                    : 'Score the data first, then filter out unstable samples.'}
                            </Paragraph>
                        </div>
                    </div>

                    {/* 适用场景 */}
                    <div className="mb-4 rounded-2xl border border-white/70 bg-white/80 p-4">
                        <div className="flex items-center gap-2 mb-3">
                            <Star className="w-4 h-4 text-purple-500" />
                            <Text strong className="text-purple-800 dark:text-purple-200">
                                {isZh ? '适用场景' : 'Use Case'}
                            </Text>
                        </div>
                        <div className="flex items-center justify-center gap-4">
                            {/* 数据源并列展示 */}
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/50 rounded-lg">
                                    <FileJson className="w-4 h-4 text-green-600" />
                                    <span className="text-sm text-green-700 dark:text-green-300">SFT</span>
                                </div>
                                <div className="flex items-center gap-2 px-4 py-2 bg-orange-100 dark:bg-orange-900/50 rounded-lg">
                                    <FileJson className="w-4 h-4 text-orange-600" />
                                    <span className="text-sm font-medium text-orange-700 dark:text-orange-300">DPO</span>
                                </div>
                            </div>
                            
                            {/* 分叉箭头 */}
                            <div className="flex flex-col items-center">
                                <ArrowRight className="w-5 h-5 text-purple-400 rotate-45" />
                                <ArrowRight className="w-5 h-5 text-purple-400 -rotate-45 -mt-2" />
                            </div>
                            
                            {/* 目标 */}
                            <div className="flex items-center gap-2 px-4 py-2 bg-purple-100 dark:bg-purple-900/50 rounded-lg">
                                <Star className="w-4 h-4 text-purple-600" />
                                <span className="text-sm text-purple-700 dark:text-purple-300">{isZh ? '高质量数据' : 'High Quality'}</span>
                            </div>
                        </div>
                    </div>

                    {/* 使用方法 */}
                    <div className="mb-4 rounded-2xl border border-slate-800 bg-slate-950 p-4">
                        <div className="flex items-center gap-2 mb-3">
                            <MessageSquare className="w-4 h-4 text-purple-400" />
                            <span className="text-sm font-medium text-purple-400">{isZh ? '对话指令' : 'Chat Command'}</span>
                        </div>
                        <div className="rounded-xl border border-slate-700 bg-slate-900/80 p-3">
                            <code className="block min-h-[44px] text-sm leading-6 text-purple-300 font-mono">
                                {'执行数据高级筛选处理，数据在<数据集路径>'}
                            </code>
                        </div>
                        <div className="mt-3 text-xs text-slate-400 space-y-1">
                            <p>• {isZh ? '输入要处理的数据路径' : 'Enter the dataset path you want to process'}</p>
                            <p>• {isZh ? '系统会自动完成打分和筛选' : 'Scoring and filtering run automatically'}</p>
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    );
}

// 步骤 3: 启动训练
function StartTrainingStep({ isZh }: { isZh: boolean }) {
    const trainingModes = [
        {
            id: 'lora-batch',
            title: isZh ? 'LoRA 批量训练' : 'LoRA Batch Training',
            icon: <Layers className="w-6 h-6" />,
            cardColor: 'blue',
            tagColor: 'green',
            command: '运行lora批量训练\n运行lora训练，学习率改为0.1，梯度累计数改为16',
            dataFormat: 'SFT',
        },
        {
            id: 'full-param-batch',
            title: isZh ? '全参批量训练' : 'Full Parameter Batch Training',
            icon: <Database className="w-6 h-6" />,
            cardColor: 'indigo',
            tagColor: 'green',
            command: '执行全参批量训练\n执行全参批量训练，学习率改为0.1，梯度累计数改为16',
            dataFormat: 'SFT',
        },
        {
            id: 'enhanced',
            title: isZh ? '增强训练' : 'Enhanced Training',
            icon: <Rocket className="w-6 h-6" />,
            cardColor: 'orange',
            tagColor: 'orange',
            command: '运行增强训练，模型位置在<模型路径>，数据集在<数据集路径>',
            dataFormat: 'DPO',
        },
        {
            id: 'grpo',
            title: isZh ? 'GRPO 训练' : 'GRPO Training',
            icon: <Activity className="w-6 h-6" />,
            cardColor: 'purple',
            tagColor: 'purple',
            command: '启动grpo训练，模型路径是<模型路径>,train_files是<训练文件路径>,val_files是<验证文件路径>，显卡是<GPU编号列表>',
            dataFormat: 'Parquet',
        },
    ];

    // 颜色类名映射
    const colorClasses: Record<string, { card: string; icon: string; badge: string }> = {
        blue: { card: 'border-sky-200/70 bg-sky-50/55', icon: 'border-sky-200/70 bg-white text-sky-600', badge: 'border-sky-200 bg-sky-50 text-sky-700' },
        indigo: { card: 'border-indigo-200/70 bg-indigo-50/55', icon: 'border-indigo-200/70 bg-white text-indigo-600', badge: 'border-indigo-200 bg-indigo-50 text-indigo-700' },
        purple: { card: 'border-violet-200/70 bg-violet-50/55', icon: 'border-violet-200/70 bg-white text-violet-600', badge: 'border-violet-200 bg-violet-50 text-violet-700' },
        orange: { card: 'border-amber-200/70 bg-amber-50/55', icon: 'border-amber-200/70 bg-white text-amber-600', badge: 'border-amber-200 bg-amber-50 text-amber-700' },
        green: { card: 'border-emerald-200/70 bg-emerald-50/55', icon: 'border-emerald-200/70 bg-white text-emerald-600', badge: 'border-emerald-200 bg-emerald-50 text-emerald-700' },
    };

    return (
        <div className="space-y-4">
            {/* 页面标题 */}
            <div className="rounded-2xl border border-border/20 bg-white/85 px-4 py-3">
                <div className="flex items-start gap-3">
                <div className="rounded-2xl border border-emerald-200/70 bg-emerald-50 p-2.5 shrink-0">
                    <Play className="w-6 h-6 text-emerald-600" />
                </div>
                <div className="flex-1">
                    <Title level={5} className="!mb-2">
                        {isZh ? '启动训练' : 'Start Training'}
                    </Title>
                    <Paragraph className="text-muted-foreground">
                        {isZh 
                            ? '选择适合当前数据和资源的模式，再把指令发到对话区即可。'
                            : 'Choose the mode that fits your data and resources, then send the command in chat.'}
                    </Paragraph>
                </div>
                </div>
            </div>

            {/* 训练类型卡片 */}
            <div className="grid grid-cols-2 gap-3">
                {trainingModes.map((mode) => {
                    const colors = colorClasses[mode.cardColor];
                    return (
                        <Card 
                            key={mode.id}
                            className={`h-full overflow-hidden rounded-2xl border ${colors.card} shadow-none`}
                            bodyStyle={{ padding: '14px', height: '100%', display: 'flex', flexDirection: 'column' }}
                        >
                            {/* 卡片头部 */}
                            <div className="flex items-start gap-3 mb-3 flex-1">
                                <div className={`rounded-2xl border p-2 shrink-0 ${colors.icon}`}>
                                    <span>{mode.icon}</span>
                                </div>
                                <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-1">
                                        <Tag className={`m-0 rounded-full ${colors.badge}`}>{mode.dataFormat}</Tag>
                                    </div>
                                    <Title level={5} className="!mb-1 !mt-0">
                                        {mode.title}
                                    </Title>
                                </div>
                            </div>

                            {/* 指令示例 */}
                            <div className="mt-auto rounded-2xl border border-slate-800 bg-slate-950 p-2.5">
                                <div className="flex items-center gap-2 mb-1.5">
                                    <MessageSquare className="w-3 h-3 text-slate-400" />
                                    <span className="text-xs text-slate-400">{isZh ? '指令示例' : 'Command Example'}</span>
                                </div>
                                <code className="block min-h-[44px] text-xs leading-5 text-green-300 font-mono break-all whitespace-pre-wrap">
                                    {mode.command}
                                </code>
                            </div>
                        </Card>
                    );
                })}
            </div>

            <Divider />

            {/* 训练模式对比说明 */}
            <div className="space-y-3">
                <Title level={5} className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-primary" />
                    {isZh ? '训练模式对比' : 'Training Mode Comparison'}
                </Title>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {/* 批量训练说明 */}
                    <Card 
                        className="rounded-2xl border border-sky-200/70 bg-sky-50/55 shadow-none"
                        bodyStyle={{ padding: '12px' }}
                    >
                        <div className="flex items-center gap-2 mb-3">
                            <Layers className="w-5 h-5 text-blue-500" />
                            <Text strong className="text-blue-800 dark:text-blue-200">
                                {isZh ? '批量训练模式' : 'Batch Training Mode'}
                            </Text>
                        </div>
                        
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <Tag color="green" className="m-0">SFT</Tag>
                                <span className="text-sm text-muted-foreground">{isZh ? '数据格式' : 'Data Format'}</span>
                            </div>
                            
                            <div className="rounded-2xl border border-white/70 bg-white/80 p-3">
                                <div className="text-sm text-blue-800 dark:text-blue-200 mb-1">
                                    {isZh ? '适用场景' : 'Use Case'}
                                </div>
                                <div className="text-xs text-blue-700 dark:text-blue-300">
                                    {isZh 
                                        ? '适用于大批量数据同时训练'
                                        : 'Suitable for large batch training'}
                                </div>
                            </div>
                        </div>
                    </Card>

                    {/* DPO 训练说明 */}
                    <Card 
                        className="rounded-2xl border border-violet-200/70 bg-violet-50/55 shadow-none"
                        bodyStyle={{ padding: '12px' }}
                    >
                        <div className="flex items-center gap-2 mb-3">
                            <Rocket className="w-5 h-5 text-purple-500" />
                            <Text strong className="text-purple-800 dark:text-purple-200">
                                {isZh ? '偏好训练模式' : 'Preference Training Mode'}
                            </Text>
                        </div>
                        
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <Tag color="orange" className="m-0">DPO</Tag>
                                <span className="text-sm text-muted-foreground">{isZh ? '数据格式' : 'Data Format'}</span>
                            </div>
                            
                            <div className="rounded-2xl border border-white/70 bg-white/80 p-3">
                                <div className="text-sm text-purple-800 dark:text-purple-200 mb-1">
                                    {isZh ? '适用场景' : 'Use Case'}
                                </div>
                                <div className="text-xs text-purple-700 dark:text-purple-300">
                                    {isZh 
                                        ? '适用于小规模、符合偏好、人类标注过的数据集'
                                        : 'Suitable for small-scale, preference-aligned, human-annotated datasets'}
                                </div>
                            </div>
                        </div>
                    </Card>

                    {/* GRPO 训练说明 */}
                    <Card 
                        className="rounded-2xl border border-fuchsia-200/70 bg-fuchsia-50/55 shadow-none"
                        bodyStyle={{ padding: '12px' }}
                    >
                        <div className="flex items-center gap-2 mb-3">
                            <Activity className="w-5 h-5 text-fuchsia-500" />
                            <Text strong className="text-fuchsia-800 dark:text-fuchsia-200">
                                {isZh ? 'GRPO 强化训练' : 'GRPO Reinforcement Training'}
                            </Text>
                        </div>
                        
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <Tag color="purple" className="m-0">Parquet</Tag>
                                <span className="text-sm text-muted-foreground">{isZh ? '数据格式' : 'Data Format'}</span>
                            </div>
                            
                            <div className="rounded-2xl border border-white/70 bg-white/80 p-3">
                                <div className="text-sm text-fuchsia-800 dark:text-fuchsia-200 mb-1">
                                    {isZh ? '适用场景' : 'Use Case'}
                                </div>
                                <div className="text-xs text-fuchsia-700 dark:text-fuchsia-300">
                                    {isZh 
                                        ? '适用于奖励驱动的推理强化训练'
                                        : 'Suitable for reward-driven reasoning reinforcement training'}
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>

                {/* 补充说明 */}
                <div className="rounded-2xl border border-amber-200/70 bg-amber-50/80 p-4">
                    <div className="flex items-start gap-3">
                        <Lightbulb className="w-5 h-5 text-amber-500 mt-0.5" />
                        <div>
                            <Text strong className="text-amber-800 dark:text-amber-200">
                                {isZh ? '使用提示' : 'Usage Tips'}
                            </Text>
                            <div className="text-sm text-amber-700 dark:text-amber-300 mt-1 space-y-1">
                                <p>• {isZh ? '开始前记得先看 GPU 是否可用' : 'Check GPU availability before you start'}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// 步骤 4: 监控训练
function MonitorTrainingStep({ isZh }: { isZh: boolean }) {
    return (
        <div className="space-y-6">
            <div className="rounded-3xl border border-border/20 bg-white/85 px-5 py-5">
                <div className="flex items-start gap-4">
                <div className="rounded-2xl border border-amber-200/70 bg-amber-50 p-3 shrink-0">
                    <LineChart className="w-7 h-7 text-amber-600" />
                </div>
                <div>
                    <Title level={5} className="!mb-2">
                        {isZh ? '实时监控训练指标' : 'Monitor Training Metrics in Real-time'}
                    </Title>
                    <Paragraph className="text-muted-foreground">
                        {isZh 
                            ? '训练开始后，先看进程状态，再从打开监控面板观察曲线变化，必要时可询问AI意见。'
                            : 'After training starts, check the process status first, then open the monitor panel to watch the curves. Ask AI for advice when needed.'}
                    </Paragraph>
                    
                </div>
                </div>
            </div>

            {/* 交互式操作演示 */}
            <InteractiveMetricsDemo isZh={isZh} />

            <Divider />

            <div className="space-y-3">
                <Title level={5} className="flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-primary" />
                    {isZh ? '训练完成后的操作' : 'Operations After Training'}
                </Title>
                <List
                    size="small"
                    bordered
                    className="overflow-hidden rounded-2xl border-border/20 bg-background"
                dataSource={[
                        isZh ? '模型会自动保存到指定位置' : 'The model is saved automatically',
                        isZh ? '可以去模型管理里查看训练结果' : 'You can review it in Model Management',
                        isZh ? '接着可以做评测验证效果' : 'Next, run an evaluation to validate quality',
                        isZh ? '需要时再导出做推理部署' : 'Export it later for inference if needed',
                    ]}
                    renderItem={(item) => (
                        <List.Item>
                            <div className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <Text>{item}</Text>
                            </div>
                        </List.Item>
                    )}
                />
            </div>

            <Alert
                message={isZh ? '恭喜！' : 'Congratulations!'}
                description={isZh 
                    ? '您已了解完整的训练流程。现在可以开始您的第一次模型训练了！'
                    : 'You have learned the complete training workflow. Now you can start your first model training!'}
                type="success"
                showIcon
                className="rounded-2xl border-emerald-200/70 bg-emerald-50/70"
            />
        </div>
    );
}

// 交互式训练指标演示组件
function InteractiveMetricsDemo({ isZh }: { isZh: boolean }) {
    const [demoStep, setDemoStep] = useState(0);

    const demoSteps = [
        { title: isZh ? '查询状态' : 'Query Status', icon: <Search className="w-4 h-4" /> },
        { title: isZh ? '打开监控面板' : 'Open Monitor Panel', icon: <LineChart className="w-4 h-4" /> },
        { title: isZh ? '查看曲线' : 'View Curves', icon: <Activity className="w-4 h-4" /> },
        { title: isZh ? '异常监控' : 'Monitor Anomalies', icon: <AlertTriangle className="w-4 h-4" /> },
        { title: isZh ? '询问AI' : 'Ask AI', icon: <Sparkles className="w-4 h-4" /> },
    ];

    const handleNext = () => {
        if (demoStep < demoSteps.length - 1) {
            setDemoStep(prev => prev + 1);
        }
    };

    const handlePrev = () => {
        if (demoStep > 0) {
            setDemoStep(prev => prev - 1);
        }
    };

    const handleReset = () => {
        setDemoStep(0);
    };

    return (
        <div className="space-y-4">
            <Title level={5} className="flex items-center gap-2">
                <MousePointerClick className="w-5 h-5 text-primary" />
                {isZh ? '操作演示' : 'Operation Demo'}
            </Title>

            {/* 演示步骤导航 */}
            <div className="flex items-center gap-2 mb-4">
                {demoSteps.map((step, index) => (
                    <div key={index} className="flex items-center">
                        <Button
                            type={demoStep === index ? 'primary' : demoStep > index ? 'default' : 'dashed'}
                            size="small"
                            icon={step.icon}
                            onClick={() => setDemoStep(index)}
                            className={demoStep === index ? 'h-9 rounded-xl bg-primary px-3' : 'h-9 rounded-xl px-3'}
                        >
                            {step.title}
                        </Button>
                        {index < demoSteps.length - 1 && (
                            <ChevronRight className="w-4 h-4 mx-1 text-muted-foreground" />
                        )}
                    </div>
                ))}
            </div>

            {/* 演示内容区域 */}
            <Card 
                className="rounded-3xl border border-border/20 bg-white/85 shadow-none"
                bodyStyle={{ padding: '20px' }}
            >
                {demoStep === 0 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 1：通过训练任务状态栏查询最新状态' : 'Step 1: Query the latest status from the training task status bar'}</Text>
                        </div>
                        
                        <div className="flex gap-4">
                            <div className="w-56 rounded-2xl border border-border/20 bg-muted/10 p-3">
                                <div className="text-xs font-medium mb-3 text-muted-foreground">{isZh ? '训练任务状态栏' : 'Training task status bar'}</div>
                                <button 
                                    className="h-7 px-2 flex items-center gap-1 text-xs hover:bg-muted rounded-md transition-colors"
                                >
                                    <Search className="w-4 h-4" />
                                    <span>{isZh ? '查询状态' : 'Query Status'}</span>
                                </button>
                            </div>
                            
                            <div className="flex-1">
                                <div className="rounded-2xl border border-sky-200/70 bg-sky-50/70 p-5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="rounded-2xl border border-sky-200/70 bg-white p-2">
                                            <Lightbulb className="w-5 h-5 text-sky-600" />
                                        </div>
                                        <Text strong className="text-base text-sky-900">
                                            {isZh ? '操作指南' : 'Operation Guide'}
                                        </Text>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">1</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '打开面板' : 'Open Panel'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '获取当前训练进程状态' : 'Fetch current training process status'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">2</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '确认状态' : 'Confirm Status'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '确认任务仍在运行或已结束' : 'Confirm whether the task is running or finished'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 1 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 2：从训练任务状态栏打开监控面板' : 'Step 2: Open the monitor panel from the training task status bar'}</Text>
                        </div>
                        
                        <div className="flex gap-4">
                            <div className="w-56 rounded-2xl border border-border/20 bg-muted/10 p-3">
                                <div className="text-xs font-medium mb-3 text-muted-foreground">{isZh ? '训练任务状态栏' : 'Training task status bar'}</div>
                                <button 
                                    className="h-7 px-2 flex items-center gap-1 text-xs hover:bg-muted rounded-md transition-colors"
                                >
                                    <LineChart className="w-4 h-4" />
                                    <span>{isZh ? '打开监控面板' : 'Open Monitor Panel'}</span>
                                </button>
                            </div>
                            
                            <div className="flex-1">
                                <div className="rounded-2xl border border-sky-200/70 bg-sky-50/70 p-5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="rounded-2xl border border-sky-200/70 bg-white p-2">
                                            <Lightbulb className="w-5 h-5 text-sky-600" />
                                        </div>
                                        <Text strong className="text-base text-sky-900">
                                            {isZh ? '操作指南' : 'Operation Guide'}
                                        </Text>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">1</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '打开面板' : 'Open Panel'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '查看右侧弹出的指标面板' : 'View the metrics panel on the right'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-600 text-sm font-bold text-white">2</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '选择进程' : 'Select Process'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '在左侧列表选择要查看的进程' : 'Select process from left list'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 2 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 3：理解训练曲线' : 'Step 3: Understand Training Curves'}</Text>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <Card size="small" className="rounded-2xl border border-sky-200/70 bg-sky-50/55 shadow-none">
                                <div className="flex items-center gap-2 mb-2">
                                    <LineChart className="w-5 h-5 text-blue-500" />
                                    <Text strong>Loss 曲线</Text>
                                </div>
                                <Paragraph className="text-xs text-muted-foreground !mb-3">
                                    {isZh 
                                        ? '正常情况下应逐步下降，后期会更平稳。'
                                        : 'A healthy loss curve trends downward, then gradually stabilizes.'}
                                </Paragraph>
                                {/* Loss曲线示例图 - 原始与平滑对比 */}
                                <div className="rounded-2xl border border-white/70 bg-white/85 p-3 min-h-[240px]">
                                    <div className="text-xs text-muted-foreground mb-2">{isZh ? '正常Loss曲线示例（原始 vs 平滑）' : 'Normal Loss Curve Example (Raw vs Smoothed)'}</div>
                                    <svg viewBox="0 0 300 150" className="w-full h-32">
                                        {/* 坐标轴 */}
                                        <line x1="30" y1="20" x2="30" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        <line x1="30" y1="130" x2="280" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        {/* Y轴标签 */}
                                        <text x="10" y="25" className="text-[8px] fill-muted-foreground">Loss</text>
                                        <text x="10" y="75" className="text-[8px] fill-muted-foreground">↓</text>
                                        {/* X轴标签 */}
                                        <text x="270" y="145" className="text-[8px] fill-muted-foreground">Step</text>
                                        <text x="150" y="145" className="text-[8px] fill-muted-foreground">→</text>
                                        
                                        {/* 原始数据 - 灰色细线，显示波动 */}
                                        <path 
                                            d="M 30 25 L 45 35 L 60 28 L 75 42 L 90 38 L 105 55 L 120 48 L 135 68 L 150 62 L 165 78 L 180 72 L 195 88 L 210 82 L 225 95 L 240 90 L 255 102 L 270 98 L 280 105" 
                                            fill="none" 
                                            stroke="#94a3b8" 
                                            strokeWidth="1"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            opacity="0.6"
                                        />
                                        {/* 原始数据点 */}
                                        <circle cx="30" cy="25" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="45" cy="35" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="60" cy="28" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="75" cy="42" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="90" cy="38" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="105" cy="55" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="120" cy="48" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="135" cy="68" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="150" cy="62" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="165" cy="78" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="180" cy="72" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="195" cy="88" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="210" cy="82" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="225" cy="95" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="240" cy="90" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="255" cy="102" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="270" cy="98" r="2" fill="#94a3b8" opacity="0.8" />
                                        <circle cx="280" cy="105" r="2" fill="#94a3b8" opacity="0.8" />
                                        
                                        {/* 平滑数据 - 蓝色粗线 */}
                                        <path 
                                            d="M 30 30 Q 55 35, 80 50 T 130 75 T 180 95 T 230 105 T 280 110" 
                                            fill="none" 
                                            stroke="#3b82f6" 
                                            strokeWidth="2.5"
                                            strokeLinecap="round"
                                        />
                                        {/* 平滑数据点 */}
                                        <circle cx="30" cy="30" r="3" fill="#3b82f6" />
                                        <circle cx="80" cy="50" r="3" fill="#3b82f6" />
                                        <circle cx="130" cy="75" r="3" fill="#3b82f6" />
                                        <circle cx="180" cy="95" r="3" fill="#3b82f6" />
                                        <circle cx="230" cy="105" r="3" fill="#3b82f6" />
                                        <circle cx="280" cy="110" r="3" fill="#3b82f6" />
                                    </svg>
                                    {/* 图例 */}
                                    <div className="flex items-center gap-4 mt-2 text-xs">
                                        <div className="flex items-center gap-1">
                                            <div className="w-4 h-0.5 bg-slate-400 opacity-60"></div>
                                            <span className="text-muted-foreground">{isZh ? '原始' : 'Raw'}</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <div className="w-4 h-1 bg-blue-500 rounded"></div>
                                            <span className="text-muted-foreground">{isZh ? '平滑' : 'Smoothed'}</span>
                                        </div>
                                    </div>
                                </div>
                            </Card>

                            <Card size="small" className="rounded-2xl border border-emerald-200/70 bg-emerald-50/55 shadow-none">
                                <div className="flex items-center gap-2 mb-2">
                                    <Sparkles className="w-5 h-5 text-green-500" />
                                    <Text strong>Learning Rate 曲线</Text>
                                </div>
                                <Paragraph className="text-xs text-muted-foreground !mb-3">
                                    {isZh 
                                        ? '通常会先预热升高，再慢慢回落。'
                                        : 'It usually warms up first, then gradually decays.'}
                                </Paragraph>
                                {/* Learning Rate曲线示例图 */}
                                <div className="rounded-2xl border border-white/70 bg-white/85 p-3 min-h-[240px]">
                                    <div className="text-xs text-muted-foreground mb-2">{isZh ? 'Learning Rate曲线示例（预热+衰减）' : 'Learning Rate Curve Example (Warmup + Decay)'}</div>
                                    <svg viewBox="0 0 300 150" className="w-full h-32">
                                        {/* 坐标轴 */}
                                        <line x1="30" y1="20" x2="30" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        <line x1="30" y1="130" x2="280" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        {/* Y轴标签 */}
                                        <text x="8" y="25" className="text-[8px] fill-muted-foreground">LR</text>
                                        <text x="10" y="75" className="text-[8px] fill-muted-foreground">↑</text>
                                        {/* X轴标签 */}
                                        <text x="270" y="145" className="text-[8px] fill-muted-foreground">Step</text>
                                        <text x="150" y="145" className="text-[8px] fill-muted-foreground">→</text>
                                        
                                        {/* 预热阶段标注 */}
                                        <text x="70" y="145" className="text-[7px] fill-green-600">{isZh ? '预热' : 'Warmup'}</text>
                                        {/* 衰减阶段标注 */}
                                        <text x="190" y="145" className="text-[7px] fill-green-600">{isZh ? '衰减' : 'Decay'}</text>
                                        
                                        {/* Learning Rate曲线 - 先升后降 */}
                                        <path 
                                            d="M 30 120 Q 60 100, 90 40 Q 120 25, 150 35 Q 180 50, 210 80 Q 240 100, 280 115" 
                                            fill="none" 
                                            stroke="#22c55e" 
                                            strokeWidth="2.5"
                                            strokeLinecap="round"
                                        />
                                        {/* 数据点 */}
                                        <circle cx="30" cy="120" r="3" fill="#22c55e" />
                                        <circle cx="90" cy="40" r="3" fill="#22c55e" />
                                        <circle cx="150" cy="35" r="3" fill="#22c55e" />
                                        <circle cx="210" cy="80" r="3" fill="#22c55e" />
                                        <circle cx="280" cy="115" r="3" fill="#22c55e" />
                                        
                                        {/* 峰值标记 */}
                                        <circle cx="120" cy="28" r="3" fill="#16a34a" />
                                        <text x="115" y="20" className="text-[7px] fill-green-600 font-medium">max</text>
                                    </svg>
                                    {/* 图例 */}
                                    <div className="flex items-center gap-4 mt-2 text-xs">
                                        <div className="flex items-center gap-1">
                                            <div className="w-4 h-1 bg-green-500 rounded"></div>
                                            <span className="text-muted-foreground">{isZh ? '学习率' : 'Learning Rate'}</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <div className="w-2 h-2 rounded-full bg-green-600"></div>
                                            <span className="text-muted-foreground">{isZh ? '峰值' : 'Peak'}</span>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                        </div>

                        <div className="rounded-2xl border border-amber-200/70 bg-amber-50/80 p-4">
                            <div className="flex items-start gap-3">
                                <Lightbulb className="w-5 h-5 text-amber-500 mt-0.5" />
                                <div>
                                    <div className="font-medium text-amber-800 dark:text-amber-200 mb-1">
                                        {isZh ? '曲线显示选项' : 'Curve Display Options'}
                                    </div>
                                    <div className="text-sm text-amber-700 dark:text-amber-300 space-y-1">
                                        <div>• {isZh ? '原始：显示未经处理的原始数据点' : 'Raw: Show unprocessed original data points'}</div>
                                        <div>• {isZh ? '平滑：应用平滑算法减少抖动，更易观察趋势' : 'Smoothed: Apply smoothing to reduce jitter and observe trends'}</div>                                        
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 3 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 4：识别异常信号' : 'Step 4: Identify Anomaly Signals'}</Text>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            {/* Loss 异常 */}
                            <Card size="small" className="rounded-2xl border border-red-200/70 bg-red-50/55 shadow-none">
                                <div className="flex items-center gap-2 mb-2">
                                    <AlertTriangle className="w-5 h-5 text-red-500" />
                                    <Text strong className="text-red-700 dark:text-red-300">{isZh ? 'Loss 异常' : 'Loss Anomaly'}</Text>
                                </div>
                                <Paragraph className="text-xs text-muted-foreground !mb-3">
                                    {isZh 
                                        ? '突然上升或剧烈波动，通常值得马上排查。'
                                        : 'A sudden spike or violent fluctuation usually deserves immediate attention.'}
                                </Paragraph>
                                {/* 异常Loss曲线示例 */}
                                <div className="rounded-2xl border border-red-200/70 bg-white/85 p-3 min-h-[220px]">
                                    <div className="text-xs text-red-600 dark:text-red-400 mb-2">{isZh ? '异常Loss曲线示例' : 'Abnormal Loss Curve'}</div>
                                    <svg viewBox="0 0 300 150" className="w-full h-32">
                                        {/* 坐标轴 */}
                                        <line x1="30" y1="20" x2="30" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        <line x1="30" y1="130" x2="280" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        {/* Y轴标签 */}
                                        <text x="10" y="25" className="text-[8px] fill-muted-foreground">Loss</text>
                                        <text x="10" y="75" className="text-[8px] fill-muted-foreground">↓</text>
                                        {/* X轴标签 */}
                                        <text x="270" y="145" className="text-[8px] fill-muted-foreground">Step</text>
                                        <text x="150" y="145" className="text-[8px] fill-muted-foreground">→</text>
                                        {/* 异常曲线 - 震荡剧烈 */}
                                        <path 
                                            d="M 30 40 L 60 80 L 90 50 L 120 100 L 150 60 L 180 110 L 210 70 L 240 90 L 280 85" 
                                            fill="none" 
                                            stroke="#ef4444" 
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                        />
                                        {/* 异常点 */}
                                        <circle cx="120" cy="100" r="3" fill="#ef4444" />
                                        <circle cx="180" cy="110" r="3" fill="#ef4444" />
                                    </svg>
                                </div>
                            </Card>

                            {/* Loss 停滞 */}
                            <Card size="small" className="rounded-2xl border border-amber-200/70 bg-amber-50/55 shadow-none">
                                <div className="flex items-center gap-2 mb-2">
                                    <AlertTriangle className="w-5 h-5 text-orange-500" />
                                    <Text strong className="text-orange-700 dark:text-orange-300">{isZh ? 'Loss 停滞' : 'Loss Stagnation'}</Text>
                                </div>
                                <Paragraph className="text-xs text-muted-foreground !mb-3">
                                    {isZh 
                                        ? '长时间没有下降，说明训练效果可能卡住了。'
                                        : 'If it stays flat for too long, training may be stuck.'}
                                </Paragraph>
                                {/* 停滞Loss曲线示例 */}
                                <div className="rounded-2xl border border-amber-200/70 bg-white/85 p-3 min-h-[220px]">
                                    <div className="text-xs text-orange-600 dark:text-orange-400 mb-2">{isZh ? '停滞Loss曲线示例' : 'Stagnant Loss Curve'}</div>
                                    <svg viewBox="0 0 300 150" className="w-full h-32">
                                        {/* 坐标轴 */}
                                        <line x1="30" y1="20" x2="30" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        <line x1="30" y1="130" x2="280" y2="130" stroke="currentColor" strokeWidth="1" className="text-muted-foreground" />
                                        {/* Y轴标签 */}
                                        <text x="10" y="25" className="text-[8px] fill-muted-foreground">Loss</text>
                                        <text x="10" y="75" className="text-[8px] fill-muted-foreground">↓</text>
                                        {/* X轴标签 */}
                                        <text x="270" y="145" className="text-[8px] fill-muted-foreground">Step</text>
                                        <text x="150" y="145" className="text-[8px] fill-muted-foreground">→</text>
                                        {/* 停滞曲线 - 几乎水平 */}
                                        <path 
                                            d="M 30 50 L 60 52 L 90 51 L 120 53 L 150 52 L 180 51 L 210 53 L 240 52 L 280 51" 
                                            fill="none" 
                                            stroke="#f97316" 
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                        />
                                        {/* 水平线标记 */}
                                        <line x1="30" y1="52" x2="280" y2="52" stroke="#f97316" strokeWidth="1" strokeDasharray="4 2" opacity="0.5" />
                                    </svg>
                                </div>
                            </Card>
                        </div>

                        <div className="rounded-2xl border border-sky-200/70 bg-sky-50/80 p-4">
                            <div className="flex items-start gap-3">
                                <Info className="w-5 h-5 text-blue-500 mt-0.5" />
                                <div>
                                    <div className="font-medium text-blue-800 dark:text-blue-200 mb-1">
                                        {isZh ? '实时监控建议' : 'Real-time Monitoring Tips'}
                                    </div>
                                    <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                                        <div>• {isZh ? '训练初期密切关注 Loss 下降趋势，确保模型开始学习' : 'Early training: closely monitor Loss decline to ensure model is learning'}</div>
                                        <div>• {isZh ? '中期观察 Loss 是否平稳，波动是否在可接受范围' : 'Mid training: observe if Loss is stable and fluctuations are acceptable'}</div>
                                        <div>• {isZh ? '后期关注是否过拟合（Loss 持续下降但验证集表现变差）' : 'Late training: watch for overfitting (Loss decreases but validation worsens)'}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {demoStep === 4 && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-4">
                            <Badge status="processing" />
                            <Text strong>{isZh ? '步骤 5：使用 AI 分析训练曲线' : 'Step 5: Use AI to Analyze Training Curves'}</Text>
                        </div>

                        <div className="flex gap-4">
                            <div className="w-56 rounded-2xl border border-border/20 bg-muted/10 p-3">
                                <div className="text-xs font-medium mb-3 text-muted-foreground">{isZh ? '指标面板工具栏' : 'Metrics Panel Toolbar'}</div>
                                <Button 
                                    variant="ghost"
                                    size="icon"
                                    className="h-10 w-10"
                                    title={isZh ? '询问AI' : 'Ask AI'}
                                >
                                    <Sparkles className="h-5 w-5" />
                                </Button>
                            </div>
                            
                            <div className="flex-1">
                                <div className="rounded-2xl border border-violet-200/70 bg-violet-50/70 p-5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="rounded-2xl border border-violet-200/70 bg-white p-2">
                                            <Sparkles className="w-5 h-5 text-violet-600" />
                                        </div>
                                        <Text strong className="text-base text-violet-900">
                                            {isZh ? '询问AI功能' : 'Ask AI Feature'}
                                        </Text>
                                    </div>
                                    
                                    <div className="space-y-3">
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-600 text-sm font-bold text-white">1</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '点击询问ai' : 'Click Ask AI Button'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '点击面板右上角的"询问AI"按钮（闪光图标）' : 'Click the "Ask AI" button (sparkle icon) in the top right of the panel'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-600 text-sm font-bold text-white">2</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '输入您的问题（可选）' : 'Enter Your Question (Optional)'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '例如："这个曲线正常吗？"、"是否存在过拟合？"。如果不输入问题，将使用默认问题："请帮我分析这个训练曲线的趋势和是否存在异常。"' : 'e.g., "Is this curve normal?", "Is there overfitting?". If no question is entered, the default will be used: "Please analyze the trend of this training curve and whether there are any anomalies."'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-600 text-sm font-bold text-white">3</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '发送给AI分析' : 'Send to AI for Analysis'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? '点击"发送给AI"按钮，将自动截图曲线发送给AI' : 'Click "Send to AI", AI will automatically capture and analyze the current training curve'}</div>
                                            </div>
                                        </div>
                                        
                                        <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 p-3">
                                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-600 text-sm font-bold text-white">4</div>
                                            <div>
                                                <div className="font-medium text-sm">{isZh ? '查看AI分析结果' : 'View AI Analysis Results'}</div>
                                                <div className="text-xs text-muted-foreground mt-1">{isZh ? 'AI会基于当前训练数据给出专业建议，包括是否正常、如何优化等' : 'AI will provide professional advice based on current training data, including normality and optimization suggestions'}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="rounded-2xl border border-emerald-200/70 bg-emerald-50/80 p-4">
                            <div className="flex items-start gap-3">
                                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                                <div>
                                    <div className="font-medium text-green-800 dark:text-green-200 mb-1">
                                        {isZh ? '何时使用询问AI' : 'When to Use Ask AI'}
                                    </div>
                                    <div className="text-sm text-green-700 dark:text-green-300 space-y-1">
                                        <div>• {isZh ? '不确定训练曲线是否正常时' : 'When unsure if the training curve is normal'}</div>
                                        <div>• {isZh ? '发现异常但不知道如何处理时' : 'When anomalies are found but unsure how to handle'}</div>
                                        <div>• {isZh ? '需要优化建议（如学习率调整）时' : 'When needing optimization advice (e.g., LR adjustment)'}</div>
                                        <div>• {isZh ? '想了解当前训练状态和质量时' : 'When wanting to understand current training status and quality'}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </Card>

            <div className="flex items-center justify-between">
                <Button
                    size="small"
                    onClick={handlePrev}
                    disabled={demoStep === 0}
                    icon={<ChevronLeft className="w-4 h-4" />}
                    className="h-9 rounded-xl px-3"
                >
                    {isZh ? '上一步' : 'Previous'}
                </Button>

                <div className="flex gap-2">
                    {demoStep === demoSteps.length - 1 ? (
                        <Button
                            size="small"
                            onClick={handleReset}
                            icon={<Sparkles className="w-4 h-4" />}
                            className="h-9 rounded-xl px-3"
                        >
                            {isZh ? '重新演示' : 'Restart'}
                        </Button>
                    ) : (
                        <Button
                            type="primary"
                            size="small"
                            onClick={handleNext}
                            icon={<ChevronRight className="w-4 h-4" />}
                            className="h-9 rounded-xl px-3"
                        >
                            {isZh ? '下一步' : 'Next'}
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}

export default TrainingWorkflowGuide;
