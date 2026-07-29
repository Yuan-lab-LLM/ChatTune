import { memo, useEffect, useRef, useState } from 'react';
import {
    InputGroup,
    InputGroupAddon,
    InputGroupButton,
    InputGroupText,
    InputGroupTextarea,
} from '@/components/ui/input-group.tsx';
import { Kbd, KbdGroup } from '@/components/ui/kbd';
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { PlayIcon, SquareIcon, Library, Save, Terminal, Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import TemplateLibraryDialog from '@/components/chat/AsChat/TemplateLibraryDialog';
import SaveTemplateDialog from '@/components/chat/AsChat/SaveTemplateDialog';
import { useTemplateLibrary, UnifiedTemplate } from '@/hooks/useTemplateLibrary';
import {
    getExampleTemplates,
    rotateExamples,
} from '@/components/chat/AsChat/template-library-examples';
import { BlockType, ContentBlocks, TextBlock } from '@shared/types';
import {
    AttachData,
    AttachItem,
} from '@/components/chat/AsChat/attach.tsx';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area.tsx';
import {
    parseSlashCommand,
} from '@/hooks/useNaturalLanguageCommands';
import { useAuth } from '@/context/AuthContext';

export interface AsTextareaProps {
    inputText?: string;
    onChange?: (text: string) => void;
    attachment?: AttachData[];
    onAttachChange?: (
        updateFn: (prevAttachData: AttachData[]) => AttachData[],
    ) => void;
    placeholder: string;
    actionType: 'send' | 'interrupt';
    onActionClick: (
        blocksInput: ContentBlocks,
        structuredInput: Record<string, unknown> | null,
    ) => void;
    disableSendBtn: boolean;
    tooltips: {
        expandTextarea?: string;
        attachButton: string;
        sendButton: string;
        interruptButton?: string;
    };
    expandable?: boolean;
    attachAccept: string[];
    attachMaxFileSize: number;
    onError: (error: string) => void;
    enableTemplateLibrary?: boolean;
    enableExampleRotation?: boolean;
    exampleRotationInterval?: number;
    // Natural language command support
    onCommand?: (input: string) => Promise<boolean>;
    enableCommandDetection?: boolean;
    // Callback when user clicks send button
    onSendComplete?: () => void;
    // Incrementing token to trigger a one-time send button highlight
    sendButtonHighlightToken?: number;
    inlineHintText?: string;
    statusHintText?: string;
    [key: string]: unknown;
}

const AsTextarea = ({
    inputText: externalInputText,
    onChange,
    attachment: externalAttachment,
    onAttachChange,
    placeholder,
    actionType,
    onActionClick,
    disableSendBtn,
    tooltips,
    expandable,
    attachAccept,
    attachMaxFileSize,
    onError,
    enableTemplateLibrary = true,
    enableExampleRotation = true,
    exampleRotationInterval = 15000,
    onCommand,
    enableCommandDetection = true,
    onSendComplete,
    sendButtonHighlightToken = 0,
    inlineHintText,
    statusHintText,
    ...props
}: AsTextareaProps) => {
    const { t } = useTranslation();
    const { isAdmin } = useAuth();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [internalInputText, setInternalInputText] = useState<string>('');
    const [internalAttachment, setInternalAttachment] = useState<AttachData[]>(
        [],
    );
    
    // Command detection state
    const [isCommandDetected, setIsCommandDetected] = useState(false);
    const [isSendButtonHighlighted, setIsSendButtonHighlighted] =
        useState(false);

    const inputText = externalInputText ?? internalInputText;

    // Example rotation state
    const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
    const examples = getExampleTemplates(isAdmin);

    // Check if user input is needed
    const isInputRequired = actionType === 'send' && !disableSendBtn;
    const isInterruptAction = actionType === 'interrupt';
    const disabledInputHint =
        t('hint.no-running-session') ||
        '未找到运行中的会话，请先启动运行实例或选择可用运行实例。';
    const disabledLibraryButtonClassName =
        'cursor-not-allowed border border-slate-200/70 bg-slate-100/90 text-slate-300 hover:bg-slate-100/90 hover:text-slate-300 dark:border-slate-700/70 dark:bg-slate-800/90 dark:text-slate-600 dark:hover:bg-slate-800/90 dark:hover:text-slate-600';

    useEffect(() => {
        if (!sendButtonHighlightToken) {
            return;
        }

        setIsSendButtonHighlighted(true);
    }, [sendButtonHighlightToken]);

    useEffect(() => {
        if (inputText.trim().length === 0) {
            setIsSendButtonHighlighted(false);
        }
    }, [inputText]);

    // Rotate examples
    useEffect(() => {
        if (!enableExampleRotation || examples.length === 0 || inputText || !isInputRequired) {
            return;
        }

        const interval = setInterval(() => {
            setCurrentExampleIndex((prev) => (prev + 1) % examples.length);
        }, exampleRotationInterval);

        return () => clearInterval(interval);
    }, [enableExampleRotation, examples.length, exampleRotationInterval, inputText, isInputRequired]);

    // Get current placeholder
    const currentPlaceholder =
        enableExampleRotation && !inputText && examples.length > 0 && isInputRequired
            ? `${placeholder} - ${rotateExamples(examples, currentExampleIndex)}`
            : placeholder;

    // 模板库状态
    const {
        isLoaded,
        getUnifiedCategories,
        getRecentTemplates,
        addToRecent,
        addCustomCategory,
        addTemplate,
        updateTemplate,
        deleteTemplate,
        getIconComponent,
        customCategories,
        userTemplates,
    } = useTemplateLibrary({ isAdmin });
    const [isLibraryOpen, setIsLibraryOpen] = useState(false);
    const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
    const [editingTemplate, setEditingTemplate] = useState<UnifiedTemplate | null>(null);
    const quickCommands = [
        {
            label: t('quickCommands.dataPreprocess') || '数据预处理',
            content: '执行数据预处理操作',
        },
        {
            label: t('quickCommands.loraBatchTrain') || 'LoRA训练',
            content: '运行lora批量训练',
        },
        {
            label: t('quickCommands.singleModelEval') || '单模型评估',
            content: '执行单模型评估',
        },
        {
            label: t('quickCommands.inferenceConfig') || '推理配置',
            content: '查看推理配置文件',
        },
        {
            label: t('quickCommands.startInference') || '启动推理',
            content: '启动推理服务',
        },
        {
            label: t('quickCommands.stopInference') || '关闭推理',
            content: '关闭推理服务',
        },
    ];

    const handleChange = (text: string) => {
        if (externalInputText === undefined) {
            setInternalInputText(text);
        }
        onChange?.(text);
        
        // Detect if input is a potential command
        if (enableCommandDetection && text.trim()) {
            const detected = parseSlashCommand(text) !== null;
            setIsCommandDetected(detected);
        } else {
            setIsCommandDetected(false);
        }
    };

    const attachment = externalAttachment ?? internalAttachment;
    const handleAttachChange = (
        updateFn: (prevAttachData: AttachData[]) => AttachData[],
    ) => {
        // If external attachment is not provided, update internal state
        if (externalAttachment === undefined) {
            setInternalAttachment(updateFn);
        }
        // Call the external handler
        onAttachChange?.(updateFn);
    };

    const handleActionClick = async () => {
        if (disableSendBtn && !isInterruptAction) {
            onError('No input is required');
            return;
        }

        setIsSendButtonHighlighted(false);

        if (actionType === 'send') {
            if (inputText.length === 0) {
                onError('No input to send');
                return;
            }
            
            // Check if this is a command and process it
            if (enableCommandDetection && onCommand) {
                const slashCommand = parseSlashCommand(inputText);
                const isPotentialCmd = slashCommand !== null;
                // 同步检查是否为命令，如果是则立即清空输入框
                if (isPotentialCmd) {
                    // 立即清空输入框，提升用户体验
                    handleChange('');
                    setIsCommandDetected(false);
                    handleAttachChange(() => []);
                    // 异步执行命令，不等待完成
                    onCommand(inputText);
                    // 通知发送完成
                    onSendComplete?.();
                    return;
                }
            }

            // Prepare the input blocks
            const blocksInput: ContentBlocks = [];
            blocksInput.push({
                type: BlockType.TEXT,
                text: inputText,
            } as TextBlock);
            blocksInput.push(...attachment.map((data) => data.block));

            // send the input
            onActionClick(blocksInput, null);

            // Clear the input
            handleChange('');
            setIsCommandDetected(false);

            // Clear the attachment
            handleAttachChange(() => []);

            // 通知发送完成
            onSendComplete?.();
        } else {
            onActionClick([], null);
        }
    };

    return (
        <div className="w-full">
            <div className="px-5 pb-2">
                <div className="mb-2 flex items-center gap-3">
                    <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400 dark:text-slate-500">
                        {t('quickCommands.title') || '常用命令'}
                    </span>
                    <div className="h-px flex-1 bg-gradient-to-r from-slate-200/80 to-transparent dark:from-white/10 dark:to-transparent" />
                </div>
                <ScrollArea className="w-full overflow-y-hidden">
                    <ScrollBar className="hidden" orientation="horizontal" />
                    <div className="flex min-w-max items-center gap-2 pr-1 pb-1">
                        {quickCommands.map((command) => (
                            <button
                                key={command.label}
                                type="button"
                                disabled={!isInputRequired}
                                className="rounded-full border border-slate-200/80 bg-white/88 px-3 py-1.5 text-[11px] font-medium text-slate-600 transition-all hover:border-sky-200 hover:bg-sky-50 hover:text-sky-700 disabled:cursor-not-allowed disabled:border-slate-200/60 disabled:bg-slate-100/70 disabled:text-slate-400 dark:border-white/10 dark:bg-slate-900/55 dark:text-slate-300 dark:hover:border-sky-500/30 dark:hover:bg-sky-500/12 dark:hover:text-sky-300 dark:disabled:border-white/10 dark:disabled:bg-slate-800/45 dark:disabled:text-slate-500"
                                onClick={() => handleChange(command.content)}
                            >
                                {command.label}
                            </button>
                        ))}
                    </div>
                </ScrollArea>
            </div>
            <div className="space-y-2">
            {statusHintText && (
                <div className="flex items-center gap-2 rounded-2xl border border-sky-200/90 bg-sky-50/90 px-4 py-2 text-[13px] text-sky-800 shadow-[0_10px_24px_-22px_rgba(14,165,233,0.7)] dark:border-sky-500/20 dark:bg-sky-500/10 dark:text-sky-200">
                    <Sparkles className="h-3.5 w-3.5 shrink-0" />
                    <span>{statusHintText}</span>
                </div>
            )}
            {!statusHintText && inlineHintText && isInputRequired && inputText.trim().length > 0 && (
                <div className="flex items-center gap-2 rounded-2xl border border-sky-200/90 bg-sky-50/90 px-4 py-2 text-[13px] text-sky-800 shadow-[0_10px_24px_-22px_rgba(14,165,233,0.7)] dark:border-sky-500/20 dark:bg-sky-500/10 dark:text-sky-200">
                    <Sparkles className="h-3.5 w-3.5 shrink-0" />
                    <span>{inlineHintText}</span>
                </div>
            )}
            <InputGroup
                className="group h-fit min-w-fit overflow-hidden rounded-[22px] border border-slate-200/90 bg-[linear-gradient(180deg,rgba(255,255,255,0.98)_0%,rgba(248,250,252,0.98)_100%)] shadow-[0_20px_44px_-34px_rgba(15,23,42,0.24)] transition-all duration-200 has-[[data-slot=input-group-control]:focus-visible]:border-slate-300 has-[[data-slot=input-group-control]:focus-visible]:shadow-[0_14px_34px_-28px_rgba(15,23,42,0.16)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.96)_0%,rgba(17,24,39,0.96)_100%)] dark:shadow-[0_20px_44px_-34px_rgba(2,6,23,0.86)] dark:has-[[data-slot=input-group-control]:focus-visible]:border-slate-600 dark:has-[[data-slot=input-group-control]:focus-visible]:shadow-[0_14px_34px_-28px_rgba(2,6,23,0.82)]"
                {...props}
            >
            <InputGroupAddon
                className={`flex flex-row h-fit w-full ${attachment.length <= 0 ? 'hidden' : ''}`}
            >
                <ScrollArea className="w-full h-fit overflow-y-hidden">
                    <ScrollBar className="hidden" orientation="horizontal" />
                    <div className="flex items-center gap-x-2 h-18">
                        {attachment.map((data, index) => (
                            <AttachItem
                                {...data}
                                onDelete={() => {
                                    handleAttachChange(
                                        (prevAttachData: AttachData[]) =>
                                            prevAttachData.filter(
                                                (_, i) => i !== index,
                                            ),
                                    );
                                }}
                            />
                        ))}
                    </div>
                </ScrollArea>
            </InputGroupAddon>
            <InputGroupTextarea
                value={inputText}
                placeholder={currentPlaceholder}
                className="min-h-[48px] px-5 py-3 text-[15px] leading-relaxed text-slate-900 placeholder:text-[12px] placeholder:leading-5 placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
                onChange={(e) => handleChange(e.target.value)}
                onKeyDown={(e) => {
                    // When typing Chinese using IME, do not trigger enter key actions
                    if (e.nativeEvent.isComposing) {
                        return;
                    }

                    // shift + enter for newline
                    if (e.key === 'Enter' && e.shiftKey) {
                        // Add a newline to the current cursor position
                        handleChange(inputText + '\n');
                        e.preventDefault();
                        return;
                    }
                    // When enter is pressed without shift, ctrl, alt, or meta, send the message
                    if (
                        e.key === 'Enter' &&
                        !e.shiftKey &&
                        !e.ctrlKey &&
                        !e.altKey &&
                        !e.metaKey
                    ) {
                        if (actionType === 'send') {
                            void handleActionClick();
                        }
                        e.preventDefault();
                        return;
                    }
                }}
            />
            <InputGroupAddon align="block-end" className="px-3 pb-2">
                <div className="flex w-full items-center justify-between px-1 py-0">
                    <InputGroupText className="text-muted-foreground/65 hidden group-focus-within:inline-flex truncate text-[11px] dark:text-slate-400">
                        <Kbd className="text-[10px]">⏎</Kbd>
                        <span className="mx-1">发送</span>
                        <span className="text-muted-foreground/40 mx-1">·</span>
                        <KbdGroup>
                            <Kbd className="text-[10px]">Shift</Kbd>
                        </KbdGroup>
                        <span className="mx-1">换行</span>
                    </InputGroupText>
                    
                    <div className="flex items-center gap-1 ml-auto">
                        {/* Command detected indicator */}
                        {enableCommandDetection && isCommandDetected && (
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <InputGroupButton
                                        variant="ghost"
                                        className="rounded-2xl text-sky-700 h-8 w-8 hover:bg-sky-50 dark:text-sky-300 dark:hover:bg-sky-500/12"
                                        size="icon-sm"
                                    >
                                        <Terminal className="w-4 h-4" />
                                    </InputGroupButton>
                                </TooltipTrigger>
                                <TooltipContent>
                                    检测到命令，按回车执行
                                </TooltipContent>
                            </Tooltip>
                        )}
                
                        {enableTemplateLibrary && isLoaded && (
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <InputGroupButton
                                        variant="ghost"
                                        className={`rounded-2xl template-library-button h-8 w-8 ${
                                            isInputRequired
                                                ? 'text-slate-600 hover:bg-sky-50 hover:text-sky-700 dark:text-slate-300 dark:hover:bg-sky-500/12 dark:hover:text-sky-300'
                                                : disabledLibraryButtonClassName
                                        }`}
                                        size="icon-sm"
                                        disabled={!isInputRequired}
                                        onClick={() => setIsLibraryOpen(true)}
                                    >
                                        <Library className="w-4 h-4" />
                                    </InputGroupButton>
                                </TooltipTrigger>
                                <TooltipContent>
                                    {isInputRequired
                                        ? (t('library') || '模板库')
                                        : disabledInputHint}
                                </TooltipContent>
                            </Tooltip>
                        )}

                        {/* 保存当前输入到模板库按钮 */}
                        {isInputRequired && inputText.trim().length > 0 && (
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <InputGroupButton
                                        variant="ghost"
                                        className="rounded-2xl h-8 w-8 text-slate-600 hover:bg-sky-50 hover:text-sky-700 dark:text-slate-300 dark:hover:bg-sky-500/12 dark:hover:text-sky-300"
                                        size="icon-sm"
                                        onClick={() => {
                                            setEditingTemplate(null);
                                            setIsSaveDialogOpen(true);
                                        }}
                                    >
                                        <Save className="w-4 h-4" />
                                    </InputGroupButton>
                                </TooltipTrigger>
                                <TooltipContent>
                                    {t('saveCurrentInput') || '保存当前输入到模板库'}
                                </TooltipContent>
                            </Tooltip>
                        )}
                        
                        <div className="relative">
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <InputGroupButton
                                        variant="default"
                                        className={`rounded-full h-9 w-9 bg-sky-600 text-white shadow-[0_14px_24px_-16px_rgba(2,132,199,0.65)] hover:bg-sky-700 ${
                                            isSendButtonHighlighted
                                                ? 'scale-110 ring-4 ring-amber-400 ring-offset-4 ring-offset-white shadow-[0_0_0_8px_rgba(251,191,36,0.28),0_18px_36px_-14px_rgba(2,132,199,0.75)] animate-[pulse_1.1s_ease-in-out_infinite] dark:ring-amber-300 dark:ring-offset-slate-900'
                                                : ''
                                        }`}
                                        size="icon-sm"
                                        disabled={disableSendBtn && !isInterruptAction}
                                        onClick={() => {
                                            handleActionClick();
                                        }}
                                    >
                                        {actionType === 'send' ? (
                                            <PlayIcon className="w-4 h-4" />
                                        ) : (
                                            <SquareIcon className="w-4 h-4" />
                                        )}
                                        <span className="sr-only">
                                            {isInterruptAction ? '停止生成' : 'Send'}
                                        </span>
                                    </InputGroupButton>
                                </TooltipTrigger>
                                <TooltipContent>
                                    {isInterruptAction
                                        ? tooltips.interruptButton || '停止生成'
                                        : tooltips.sendButton}
                                </TooltipContent>
                            </Tooltip>
                        </div>
                    </div>
                </div>
            </InputGroupAddon>

            {/* 模板库对话框 */}
            <TemplateLibraryDialog
                open={isLibraryOpen}
                onOpenChange={setIsLibraryOpen}
                categories={getUnifiedCategories()}
                recentTemplates={getRecentTemplates()}
                currentInput={inputText}
                onInsert={(content) => {
                    handleChange(content);
                }}
                onSaveNew={() => {
                    setEditingTemplate(null);
                    setIsSaveDialogOpen(true);
                }}
                onEdit={(template) => {
                    setEditingTemplate(template);
                    setIsSaveDialogOpen(true);
                }}
                onDelete={(id) => {
                    const confirmed = window.confirm(
                        t('deleteTemplateConfirm') || '确定删除这个模板吗？',
                    );
                    if (confirmed) {
                        deleteTemplate(id);
                    }
                }}
                addToRecent={addToRecent}
                getIconComponent={getIconComponent}
            />

            {/* 保存模板对话框 */}
            <SaveTemplateDialog
                open={isSaveDialogOpen}
                onOpenChange={(open) => {
                    setIsSaveDialogOpen(open);
                    if (!open) {
                        setEditingTemplate(null);
                    }
                }}
                content={editingTemplate?.content ?? inputText}
                onSave={(name, content, category) => {
                    if (editingTemplate?.source === 'user') {
                        updateTemplate(editingTemplate.id, { name, content, category });
                    } else {
                        addTemplate(name, content, category);
                    }
                    setIsSaveDialogOpen(false);
                    setEditingTemplate(null);
                }}
                customCategories={customCategories}
                onAddCategory={addCustomCategory}
                mode={editingTemplate ? 'edit' : 'create'}
                initialTemplate={
                    editingTemplate?.source === 'user'
                        ? userTemplates.find(template => template.id === editingTemplate.id) || null
                        : null
                }
                existingTemplates={userTemplates}
            />
            </InputGroup>
            </div>
        </div>
    );
};

export default memo(AsTextarea);
