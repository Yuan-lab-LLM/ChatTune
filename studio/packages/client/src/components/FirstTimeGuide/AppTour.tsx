import { Tour, message, Modal, Button, Alert } from 'antd';
import { useTranslation } from 'react-i18next';
import { useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect, useCallback, useRef } from 'react';
import { useFirstTimeGuide } from '@/context/FirstTimeGuideContext';
import { useSocket } from '@/context/SocketContext';
import { ResponseBody, RunData, SocketEvents, Status } from '@shared/types';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export function AppTour() {
    const { i18n } = useTranslation();
    const location = useLocation();
    const navigate = useNavigate();
    const socket = useSocket();
    const { showTour, isWaitingForProjectLoad, completeTour, closeTourWithoutCompleting } = useFirstTimeGuide();
    const [currentStep, setCurrentStep] = useState(0);
    const [isWaitingForRoute, setIsWaitingForRoute] = useState(false);
    const [step2Ready, setStep2Ready] = useState(false);
    const [step3Ready, setStep3Ready] = useState(false);
    const [step4Ready, setStep4Ready] = useState(false);
    const [step5Ready, setStep5Ready] = useState(false);
    const [step6Ready, setStep6Ready] = useState(false);
    const [activeTab, setActiveTab] = useState<'runs' | 'datasets' | 'models' | 'evaluation'>('runs');
    const projectNameRef = useRef<string | null>(null);
    const runCheckTokenRef = useRef(0);
    const step4ReadyTimeoutRef = useRef<number | null>(null);
    const step4StableTimerRef = useRef<number | null>(null);
    const step4ObserverRef = useRef<MutationObserver | null>(null);
    
    // 无runs时的Modal状态
    const [showNoRunsModal, setShowNoRunsModal] = useState(false);
    const [isCheckingRuns, setIsCheckingRuns] = useState(false);
    const [projectRuns, setProjectRuns] = useState<RunData[]>([]);
    const [hasReceivedProjectRuns, setHasReceivedProjectRuns] = useState(false);
    
    // Step 3 子步骤状态 (0: 新手引导, 1: 对话输入栏, 2: 模板库)
    const [step3SubStep, setStep3SubStep] = useState(0);
    
    const isZh = i18n.language === 'zh';

    const getAvailableRuns = useCallback(() => {
        return projectRuns.filter((run) => {
            return run.status === Status.RUNNING || run.status === Status.PENDING;
        });
    }, [projectRuns]);

    const getPreferredRun = useCallback(() => {
        const sortByLatest = (runs: RunData[]) => {
            return [...runs].sort((a, b) => {
                return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
            });
        };

        const runningRuns = sortByLatest(
            projectRuns.filter((run) => run.status === Status.RUNNING),
        );
        if (runningRuns.length > 0) {
            return runningRuns[0];
        }

        const pendingRuns = sortByLatest(
            projectRuns.filter((run) => run.status === Status.PENDING),
        );
        return pendingRuns[0] || null;
    }, [projectRuns]);

    const getPreferredRunIdFromDom = useCallback(() => {
        const rows = Array.from(
            document.querySelectorAll('[data-run-id][data-run-status]'),
        ) as HTMLElement[];

        const availableRows = rows.filter((row) => {
            const status = (row.getAttribute('data-run-status') || '').toLowerCase();
            return status === Status.RUNNING || status === Status.PENDING;
        });

        if (availableRows.length === 0) {
            return null;
        }

        const runningRow =
            availableRows.find((row) => {
                return (row.getAttribute('data-run-status') || '').toLowerCase() === Status.RUNNING;
            }) || null;

        const preferredRow = runningRow || availableRows[0];
        const runId = preferredRow.getAttribute('data-run-id');
        return runId;
    }, []);

    const waitForAvailableRuns = useCallback(async (timeoutMs = 12000, intervalMs = 200) => {
        const startTime = Date.now();

        while (Date.now() - startTime < timeoutMs) {
            const availableRuns = getAvailableRuns();

            if (availableRuns.length > 0) {
                return true;
            }

            await new Promise((resolve) => setTimeout(resolve, intervalMs));
        }

        const availableRuns = getAvailableRuns();
        return availableRuns.length > 0;
    }, [getAvailableRuns, hasReceivedProjectRuns, projectRuns.length]);

    const activateRunPageSection = useCallback((section: 'runs' | 'datasets' | 'models' | 'evaluation') => {
        const triggerSectionButton = () => {
            const target = document.querySelector(`[data-section="${section}"]`) as HTMLButtonElement | null;
            if (!target) {
                return false;
            }

            target.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
            target.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
            target.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
            target.click();
            return true;
        };

        if (triggerSectionButton()) {
            return;
        }

        let attempts = 0;
        const maxAttempts = 15;
        const timer = window.setInterval(() => {
            attempts += 1;

            if (triggerSectionButton() || attempts >= maxAttempts) {
                window.clearInterval(timer);
            }
        }, 120);
    }, []);

    const getVisibleElement = useCallback((selector: string) => {
        const el = document.querySelector(selector) as HTMLElement | null;
        if (!el) {
            return null;
        }

        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
            return null;
        }

        return el;
    }, []);

    const clearStep4PendingWork = useCallback(() => {
        if (step4ObserverRef.current) {
            step4ObserverRef.current.disconnect();
            step4ObserverRef.current = null;
        }

        if (step4ReadyTimeoutRef.current) {
            window.clearTimeout(step4ReadyTimeoutRef.current);
            step4ReadyTimeoutRef.current = null;
        }

        if (step4StableTimerRef.current) {
            window.clearTimeout(step4StableTimerRef.current);
            step4StableTimerRef.current = null;
        }
    }, []);

    // 从当前路径解析项目名
    const projectName = location.pathname.match(/\/projects\/([^/]+)/)?.[1];
    
    // 保存项目名称引用
    useEffect(() => {
        if (projectName) {
            projectNameRef.current = projectName;
        }
    }, [projectName]);

    useEffect(() => {
        if (!socket || !projectName) {
            setProjectRuns([]);
            setHasReceivedProjectRuns(false);
            return;
        }

        const handleRunsData = (runs: RunData[]) => {
            setProjectRuns(runs);
            setHasReceivedProjectRuns(true);
        };

        socket.on(SocketEvents.server.pushRunsData, handleRunsData);
        socket.emit(
            SocketEvents.client.joinProjectRoom,
            projectName,
            (response: ResponseBody) => {
                if (!response.success) {
                    console.error('Failed to join project room:', response.message);
                }
            },
        );

        return () => {
            socket.off(SocketEvents.server.pushRunsData, handleRunsData);
        };
    }, [socket, projectName]);
    
    // 监听路由变化
    useEffect(() => {
        
        if (isWaitingForRoute && currentStep === 2) {
            // 检查是否已进入具体 run 页面 (/runs/xxx)
            const isInSpecificRun = location.pathname.match(/\/runs\/[^/]+$/);
            
            if (isInSpecificRun) {
                const closePanelButton = document.querySelector(
                    'button[aria-label="Close management panel"]',
                ) as HTMLButtonElement | null;
                if (closePanelButton) {
                    closePanelButton.click();
                }

                
                // 首先检查按钮是否已存在
                const checkTarget = () => {
                    const wrapper = document.querySelector('.quick-start-wrapper') as HTMLElement;
                    return !!wrapper;
                };
                
                // 如果按钮已存在，立即就绪
                if (checkTarget()) {
                    setStep3Ready(true);
                    setIsWaitingForRoute(false);
                    return;
                }
                
                // 使用MutationObserver监听DOM变化
                const observer = new MutationObserver((mutations) => {
                    if (checkTarget()) {
                        setStep3Ready(true);
                        setIsWaitingForRoute(false);
                        observer.disconnect();
                    }
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ['class']
                });
                
                // 短暂等待后强制就绪，避免首次进入时卡太久
                const fallbackTimer = setTimeout(() => {
                    setStep3Ready(true);
                    setIsWaitingForRoute(false);
                    observer.disconnect();
                }, 400);
                
                return () => {
                    observer.disconnect();
                    clearTimeout(fallbackTimer);
                };
            }
        }
        
        // Step 4 -> Step 5: 切换到 models tab
        if (currentStep === 4 && activeTab === 'datasets') {
            activateRunPageSection('models');
            setActiveTab('models');
            setTimeout(() => {
                setStep5Ready(true);
            }, 500);
        }
        
        // Step 5 -> Step 6: 切换到 evaluation tab
        if (currentStep === 5 && activeTab === 'models') {
            activateRunPageSection('evaluation');
            setActiveTab('evaluation');
            setTimeout(() => {
                setStep6Ready(true);
            }, 500);
        }
    }, [location, isWaitingForRoute, currentStep, activeTab, activateRunPageSection]);
    
    // Step 1 -> Step 2 时智能检测表格数据加载
    useEffect(() => {
        if (currentStep === 1 && showTour) {

            activateRunPageSection('runs');
            setActiveTab('runs');
            
            // 首先检查表格是否已存在
            const checkTable = () => {
                const table = document.querySelector('.run-sider-table .ant-table-body, .run-sider-table .ant-table-container, .run-sider-table table');
                const rows = document.querySelectorAll('.run-sider-table .ant-table-row, .run-sider-table .ant-table-row-level-0, .run-sider-table tr');
                return !!table && rows.length > 0;
            };
            
            // 如果表格已存在，立即就绪
            if (checkTable()) {
                setStep2Ready(true);
                return;
            }
            
            // 使用MutationObserver监听DOM变化
            const observer = new MutationObserver((mutations) => {
                if (checkTable()) {
                    setStep2Ready(true);
                    observer.disconnect();
                }
            });

            const ensureRunsOpenTimer = window.setInterval(() => {
                if (checkTable()) {
                    window.clearInterval(ensureRunsOpenTimer);
                    return;
                }

                activateRunPageSection('runs');
            }, 250);
            
            observer.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['class']
            });
            
            // 3秒后强制就绪（fallback）
            const fallbackTimer = setTimeout(() => {
                setStep2Ready(true);
                observer.disconnect();
                window.clearInterval(ensureRunsOpenTimer);
            }, 3000);
            
            return () => {
                observer.disconnect();
                window.clearInterval(ensureRunsOpenTimer);
                clearTimeout(fallbackTimer);
            };
        }
    }, [currentStep, showTour, activateRunPageSection]);

    // Step 2 -> Step 3 时选择 running run 并导航
    const handleStep2Next = useCallback(async () => {
        runCheckTokenRef.current += 1;
        
        if (!projectName) {
            console.error('No project name found');
            message.warning(isZh ? '无法获取项目名称' : 'Cannot get project name');
            closeTourWithoutCompleting();
            return;
        }
        
        let preferredRun = getPreferredRun();
        let runId = preferredRun?.id || getPreferredRunIdFromDom();

        if (!runId) {
            const hasRuns = await waitForAvailableRuns();
            preferredRun = hasRuns ? getPreferredRun() : null;
            runId = preferredRun?.id || getPreferredRunIdFromDom();
        }


        if (!runId) {
            console.error('No running run found');
            message.warning(isZh 
                ? '未找到运行中的会话，请检查后台程序是否运行中' 
                : 'No running session found, please check if backend is running'
            );
            closeTourWithoutCompleting();
            return;
        }

        const targetPath = `/projects/${projectName}/runs/${runId}`;
        setIsWaitingForRoute(true);
        setCurrentStep(2);
        navigate(targetPath);
    }, [isZh, closeTourWithoutCompleting, getPreferredRun, getPreferredRunIdFromDom, navigate, projectName, waitForAvailableRuns]);
    
    // Step 3 -> Step 4 时直接在当前页面切到 datasets，避免先退回 runs 再打开侧栏的卡顿
    const handleStep3Next = useCallback(() => {
        clearStep4PendingWork();
        setStep4Ready(false);
        activateRunPageSection('datasets');
        setActiveTab('datasets');
        setCurrentStep(3);

        let settled = false;

        const markStep4Ready = () => {
            if (settled) {
                return;
            }

            settled = true;
            clearStep4PendingWork();

            // Match Step 5/6 more closely by waiting a short moment for the datasets panel to finish layout.
            step4StableTimerRef.current = window.setTimeout(() => {
                step4StableTimerRef.current = null;
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        setStep4Ready(true);
                    });
                });
            }, 180);
        };

        const checkDatasetTarget = () => {
            const el =
                getVisibleElement('.dataset-query-button') ||
                getVisibleElement('[data-section="datasets"]');
            if (el) {
                markStep4Ready();
                return true;
            }
            return false;
        };

        if (checkDatasetTarget()) {
            return;
        }

        step4ObserverRef.current = new MutationObserver(() => {
            if (checkDatasetTarget()) {
                return;
            }
        });

        step4ObserverRef.current.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class'],
        });

        step4ReadyTimeoutRef.current = window.setTimeout(() => {
            clearStep4PendingWork();
            setStep4Ready(true);
        }, 3000);
    }, [activateRunPageSection, clearStep4PendingWork, getVisibleElement]);
    
    // 检查是否有可用的runs
    // Step 1 点击下一步时的处理
    const handleStep1Next = useCallback(async () => {
        setIsCheckingRuns(true);
        const token = ++runCheckTokenRef.current;

        const hasRuns = await waitForAvailableRuns();

        if (token !== runCheckTokenRef.current) {
            setIsCheckingRuns(false);
            return;
        }

        if (hasRuns) {
            activateRunPageSection('runs');
            setActiveTab('runs');
            setCurrentStep(1);
            setStep2Ready(false);
        } else {
            setShowNoRunsModal(true);
        }

        setIsCheckingRuns(false);
    }, [waitForAvailableRuns, activateRunPageSection]);
    
    // 刷新检测runs
    const handleRefreshCheck = useCallback(async () => {
        setIsCheckingRuns(true);
        const token = ++runCheckTokenRef.current;

        const hasRuns = await waitForAvailableRuns();

        if (token !== runCheckTokenRef.current) {
            setIsCheckingRuns(false);
            return;
        }

        if (hasRuns) {
            setShowNoRunsModal(false);
            activateRunPageSection('runs');
            setActiveTab('runs');
            setCurrentStep(1);
            setStep2Ready(false);
        } else {
        }

        setIsCheckingRuns(false);
    }, [waitForAvailableRuns, activateRunPageSection]);
    
    // 关闭Modal和Tour，但不标记为已完成（下次仍可进入引导）
    const handleCloseNoRunsModal = useCallback(() => {
        setShowNoRunsModal(false);
        closeTourWithoutCompleting(); // 关闭引导但不记录完成状态
    }, [closeTourWithoutCompleting]);
    
    // 重写 steps 配置（6步）
    const getSteps = useCallback(() => {
        const baseSteps = [
            {
                title: isZh ? 'Step 1: 左侧一级菜单' : 'Step 1: Primary Navigation',
                description: isZh 
                    ? '这里可以快速切换五大功能模块：运行列表、系统概览、数据管理、模型管理、评测管理。'
                    : 'Quickly switch between five modules: Run List, System Overview, Data, Models, and Evaluation.',
                target: () => document.querySelector('[data-runpage-nav="true"]') as HTMLElement,
                placement: 'right' as const,
                prevButtonProps: {
                    style: { display: 'none' }
                },
                nextButtonProps: {
                    children: isZh ? '下一步' : 'Next',
                    onClick: handleStep1Next,
                    loading: isCheckingRuns,
                }
            },
            {
                title: isZh ? 'Step 2: 选择运行实例' : 'Step 2: Select Running Instance',
                description: isZh
                    ? '在列表中选择运行中的会话，点击下一步将自动选择第一个运行中的会话并进入对话页面。'
                    : 'Select a running session from the list. Click Next to auto-select the first running session and enter the conversation.',
                target: () => {
                    
                    // 如果表格已准备好，返回表格区域
                    if (step2Ready && activeTab === 'runs') {
                        // 优先查找包含数据的表格容器
                        const tableContainer = document.querySelector('.run-sider-table .ant-table-body') as HTMLElement;
                        const table = document.querySelector('.run-sider-table .ant-table-container, .run-sider-table table') as HTMLElement;
                        const el = tableContainer || table;
                        
                        
                        if (el) {
                            return el;
                        }
                    }
                    
                    // 如果表格不存在，返回body但居中显示引导卡片
                    return document.body as HTMLElement;
                },
                placement: 'bottom' as const,
                nextButtonProps: {
                    children: isZh ? '下一步' : 'Next',
                    onClick: handleStep2Next,
                },
                prevButtonProps: {
                    children: null, // 隐藏上一步按钮
                    style: { display: 'none' }
                }
            },
            // Step 3: 对话页面功能介绍（3个子步骤）
            {
                title: (() => {
                    const titles = [
                        isZh ? 'Step 3-1: 新手引导' : 'Step 3-1: Beginner Guide',
                        isZh ? 'Step 3-2: 对话输入' : 'Step 3-2: Chat Input',
                        isZh ? 'Step 3-3: 模板库' : 'Step 3-3: Template Library'
                    ];
                    return titles[step3SubStep];
                })(),
                description: (() => {
                    const descriptions = [
                        isZh
                            ? '点击顶部"新手引导"可选择"学习教程"或"实战演练"：学习教程适合先理解流程，实战演练会带您按步骤跑一遍示例。'
                            : 'Click "Beginner Guide" to choose Learning Tutorial or Practice Walkthrough. The tutorial explains the workflow; the walkthrough guides you through a sample run step by step.',
                        isZh
                            ? '在输入框中输入您的问题或指令，与 AI 进行对话。'
                            : 'Type your questions or instructions in the input box to chat with AI.',
                        isZh
                            ? '点击"模板库"按钮（书本图标）查看预设指令模板。'
                            : 'Click the "Template Library" button (book icon) to view preset instruction templates.'
                    ];
                    return descriptions[step3SubStep];
                })(),
                target: () => {
                    
                    if (!step3Ready) {
                        return document.body as HTMLElement;
                    }
                    
                    // 根据子步骤返回不同的目标元素
                    let el: HTMLElement | null = null;
                    
                    switch (step3SubStep) {
                        case 0:
                            el = document.querySelector('.quick-start-wrapper') as HTMLElement;
                            break;
                        case 1:
                            el = document.querySelector('.chat-input-area') as HTMLElement;
                            break;
                        case 2:
                            el = document.querySelector('.template-library-button') as HTMLElement;
                            break;
                    }
                    
                    return el || document.body as HTMLElement;
                },
                placement: (() => {
                    const placements: Array<'bottom' | 'top' | 'left' | 'right'> = ['bottom', 'top', 'top'];
                    return placements[step3SubStep];
                })(),
                gap: (() => {
                    const gaps = [
                        { offset: 8, radius: 4 },  // Step 3-1: 新手引导，默认偏移
                        { offset: 8, radius: 4 },  // Step 3-2: 对话输入，默认偏移
                        { offset: 8, radius: 4 }   // Step 3-3: 模板库，默认偏移
                    ];
                    return gaps[step3SubStep];
                })(),
                nextButtonProps: {
                    children: (() => {
                        // 最后一个子步骤显示"下一步"，其他显示"继续"
                        return step3SubStep === 2
                            ? (isZh ? '下一步' : 'Next') 
                            : (isZh ? '继续' : 'Continue');
                    })(),
                    onClick: () => {
                        if (step3SubStep < 2) {
                            // 在Step 3内部循环
                            setStep3SubStep(prev => prev + 1);
                        } else {
                            // 最后一个子步骤，进入Step 4
                            handleStep3Next();
                            setStep3SubStep(0); // 重置子步骤
                        }
                    },
                },
                prevButtonProps: {
                    style: { display: 'none' }
                }
            },
            {
                title: isZh ? 'Step 4: 数据管理' : 'Step 4: Data Management',
                description: isZh
                    ? '点击左侧一级菜单中的"数据管理"，再点击"查询可用数据集"按钮，输入容器名称即可查询可用数据集。'
                    : 'Open Data Management from the primary navigation, then click "Query Available Datasets" and enter a container name.',
                target: () => {
                    if (!step4Ready || activeTab !== 'datasets') {
                        return document.body as HTMLElement;
                    }
                    const el =
                        getVisibleElement('.dataset-query-button') ||
                        getVisibleElement('[data-section="datasets"]');
                    return el || document.body as HTMLElement;
                },
                placement: 'right' as const,
                prevButtonProps: {
                    style: { display: 'none' }
                }
            },
            {
                title: isZh ? 'Step 5: 模型管理' : 'Step 5: Model Management',
                description: isZh
                    ? '点击左侧一级菜单中的"模型管理"，再点击"查询可用模型"按钮，输入容器名称即可查询可用模型。'
                    : 'Open Model Management from the primary navigation, then click "Query Available Models" and enter a container name.',
                target: () => {
                    if (!step5Ready || activeTab !== 'models') {
                        return document.body as HTMLElement;
                    }
                    const el = getVisibleElement('.model-query-button');
                    return el || document.body as HTMLElement;
                },
                placement: 'right' as const,
                prevButtonProps: {
                    style: { display: 'none' }
                }
            },
            {
                title: isZh ? 'Step 6: 评测管理' : 'Step 6: Evaluation Management',
                description: isZh
                    ? '点击左侧一级菜单中的"评测管理"，再点击"查询可用评测集"按钮，输入容器名称即可查询可用评测集。'
                    : 'Open Evaluation Management from the primary navigation, then click "Query Available Tests" and enter a container name.',
                target: () => {
                    if (!step6Ready || activeTab !== 'evaluation') {
                        return document.body as HTMLElement;
                    }
                    const el = getVisibleElement('.evaluation-query-button');
                    return el || document.body as HTMLElement;
                },
                placement: 'right' as const,
                prevButtonProps: {
                    style: { display: 'none' }
                }
            },
        ];
        
        // 如果正在等待路由，只显示前3步
        if (isWaitingForRoute) {
            return baseSteps.slice(0, 3);
        }
        
        return baseSteps;
    }, [isZh, step2Ready, step3Ready, step4Ready, step5Ready, step6Ready, isWaitingForRoute, activeTab, handleStep1Next, handleStep2Next, handleStep3Next, isCheckingRuns, step3SubStep, getVisibleElement]);
    
    const steps = getSteps();
    
    // 当进入各步骤时，确保相应状态就绪
    useEffect(() => {
        
        if (currentStep === 2 && step3Ready) {
            const timer = setTimeout(() => {
                setCurrentStep(2);
            }, 100);
            return () => clearTimeout(timer);
        }
        
        if (currentStep === 3 && step4Ready) {
            const timer = setTimeout(() => {
                setCurrentStep(3);
            }, 100);
            return () => clearTimeout(timer);
        }
        
        if (currentStep === 4 && step5Ready) {
            const timer = setTimeout(() => {
                setCurrentStep(4);
            }, 100);
            return () => clearTimeout(timer);
        }
        
        if (currentStep === 5 && step6Ready) {
            const timer = setTimeout(() => {
                setCurrentStep(5);
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [currentStep, step3Ready, step4Ready, step5Ready, step6Ready]);
    
    // 自动切换左侧一级菜单
    useEffect(() => {
        if (currentStep === 3 && activeTab !== 'datasets') {
            const datasetsBtn = document.querySelector('button[data-section="datasets"]') as HTMLElement;
            if (datasetsBtn) {
                datasetsBtn.click();
            }
        }
        if (currentStep === 4 && activeTab !== 'models') {
            const modelsBtn = document.querySelector('button[data-section="models"]') as HTMLElement;
            if (modelsBtn) {
                modelsBtn.click();
            }
        }
        if (currentStep === 5 && activeTab !== 'evaluation') {
            const evaluationBtn = document.querySelector('button[data-section="evaluation"]') as HTMLElement;
            if (evaluationBtn) {
                evaluationBtn.click();
            }
        }
    }, [currentStep, activeTab]);
    
    // 引导关闭时重置所有状态
    useEffect(() => {
        if (currentStep >= 2) {
            runCheckTokenRef.current += 1;
        }
    }, [currentStep]);

    useEffect(() => {
        if (!showTour) {
            clearStep4PendingWork();
            runCheckTokenRef.current += 1;
            setStep3SubStep(0);
            setCurrentStep(0);
            setStep2Ready(false);
            setStep3Ready(false);
            setStep4Ready(false);
            setStep5Ready(false);
            setStep6Ready(false);
            setActiveTab('runs');
            setIsWaitingForRoute(false);
        }
    }, [clearStep4PendingWork, showTour]);
    
    // 如果引导未开启，不渲染任何内容
    if (!showTour) return null;
    
    // Modal 内容
    const noRunsModalContent = {
        title: isZh ? (
            <span className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                未检测到运行中的会话
            </span>
        ) : (
            <span className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                No Running Sessions Detected
            </span>
        ),
        description: isZh 
            ? '引导需要至少一个运行中的会话（Running状态）才能继续演示。'
            : 'The tour requires at least one running session (Running status) to continue.',
        reasons: isZh ? [
            '• 后端程序可能未启动',
            '• 检查端口号是否已更改'
        ] : [
            '• Backend may not be running',
            '• Check if the port has been changed'
        ],
        solutions: isZh ? [
            '1. 启动后端程序，运行后刷新页面',
            '2. 检查端口号后重试'
        ] : [
            '1. Start the backend program and refresh',
            '2. Check the port and try again'
        ],
        primaryButton: isZh ? '稍后再试' : 'Try Again Later',
        secondaryButton: isZh ? '刷新检测' : 'Refresh Check',
        helpText: isZh 
            ? '提示：启动后端程序后，点击"刷新检测"按钮可以继续引导。点击"稍后再试"将关闭引导，下次进入后将重新进入引导。' 
            : 'Tip: Click "Refresh Check" to continue after starting the backend. Click "Try Again Later" to close the tour and re-enter next time.',
    };
    
    // 确定 Tour 是否显示：页面加载完成且不显示警告弹窗
    const shouldShowTour = showTour && !isWaitingForProjectLoad && !showNoRunsModal;
    
    return (
        <>
            {/* 页面加载中提示 */}
            {isWaitingForProjectLoad && (
                <div className="fixed inset-0 z-[1001] flex items-center justify-center bg-black/20">
                    <div className="bg-white rounded-lg p-6 shadow-lg max-w-sm">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                            <span className="text-lg font-medium">
                                {isZh ? '正在加载...' : 'Loading...'}
                            </span>
                        </div>
                        <p className="text-muted-foreground text-sm">
                            {isZh 
                                ? '正在加载项目数据，请稍候...' 
                                : 'Loading project data, please wait...'}
                        </p>
                    </div>
                </div>
            )}
            
            <Tour
                open={shouldShowTour}
                onClose={() => {
                    if (currentStep >= 5) {
                        completeTour();
                    } else {
                        closeTourWithoutCompleting();
                    }
                }}
                onChange={(step) => {

                    // Step 3 有 3 个子步骤，完全由 step3SubStep 驱动，忽略 Tour 默认的大步骤推进
                    if (currentStep === 2 && step >= 3) {
                        return;
                    }

                    // 允许步骤正常变化，包括从step 2前进到其他步骤
                    if (step !== currentStep) {
                        setCurrentStep(step);
                    }
                }}
                current={currentStep}
                steps={steps}
                mask={true}
                type="primary"
                showPrev={false}
            />
            
            {/* 无 Runs 时的提示 Modal */}
            <Modal
                title={noRunsModalContent.title}
                open={showNoRunsModal}
                onCancel={handleCloseNoRunsModal}
                footer={null}
                closable={false}
                maskClosable={false}
                centered
                width={480}
            >
                <div className="space-y-4">
                    <Alert
                        message={noRunsModalContent.description}
                        type="warning"
                        showIcon
                    />
                    
                    <div>
                        <h4 className="font-semibold text-foreground text-sm mb-2">
                            {isZh ? '可能的原因：' : 'Possible Reasons:'}
                        </h4>
                        <div className="text-sm text-muted-foreground/80 space-y-1 pl-1">
                            {noRunsModalContent.reasons.map((reason, index) => (
                                <p key={index}>{reason}</p>
                            ))}
                        </div>
                    </div>
                    
                    <div>
                        <h4 className="font-semibold text-foreground text-sm mb-2">
                            {isZh ? '解决方案：' : 'Solutions:'}
                        </h4>
                        <div className="text-sm text-muted-foreground/80 space-y-1 pl-1">
                            {noRunsModalContent.solutions.map((solution, index) => (
                                <p key={index}>{solution}</p>
                            ))}
                        </div>
                    </div>
                    
                    <p className="text-xs text-muted-foreground/70 text-center">
                        {noRunsModalContent.helpText}
                    </p>
                    
                    <div className="pt-4 border-t border-border">
                        <div className="flex items-center justify-end gap-2">
                            <Button 
                                onClick={handleRefreshCheck}
                                loading={isCheckingRuns}
                                icon={<RefreshCw className="w-4 h-4" />}
                            >
                                {noRunsModalContent.secondaryButton}
                            </Button>
                            <Button 
                                type="primary" 
                                onClick={handleCloseNoRunsModal}
                            >
                                {noRunsModalContent.primaryButton}
                            </Button>
                        </div>
                    </div>
                </div>
            </Modal>
        </>
    );
}
