import { useState, useMemo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import {
    Search,
    Edit2,
    Trash2,
    Plus,
    Clock,
    BookOpen,
    User,
} from 'lucide-react';
import {
    UnifiedTemplate,
    UnifiedCategory,
} from '@/hooks/useTemplateLibrary';

interface TemplateLibraryDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    categories: UnifiedCategory[];
    recentTemplates: UnifiedTemplate[];
    currentInput: string;
    onInsert: (content: string) => void;
    onSaveNew: () => void;
    onEdit?: (template: UnifiedTemplate) => void;
    onDelete?: (id: string) => void;
    addToRecent?: (templateId: string) => void;
    getIconComponent: (iconName: string) => React.ComponentType<{ className?: string }> | undefined;
    initialCategory?: string | null;
}

const TemplateLibraryDialog = ({
    open,
    onOpenChange,
    categories,
    recentTemplates,
    currentInput,
    onInsert,
    onSaveNew,
    onEdit,
    onDelete,
    addToRecent,
    getIconComponent,
    initialCategory = null,
}: TemplateLibraryDialogProps) => {
    const { t } = useTranslation();
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<string | null>(initialCategory);

    const translateCategoryName = (categoryIdOrName: string) => {
        const categoryKeyMap: Record<string, string> = {
            Beginner: 'categoryBeginner',
            beginner: 'categoryBeginner',
            Data: 'categoryData',
            training: 'categoryTraining',
            Training: 'categoryTraining',
            train: 'categoryTraining',
            assessment: 'categoryAssessment',
            Assessment: 'categoryAssessment',
            evaluation: 'categoryEvaluation',
            Evaluation: 'categoryEvaluation',
            inference: 'categoryInference',
            Inference: 'categoryInference',
            data: 'categoryData',
            management: 'categoryManagement',
            Management: 'categoryManagement',
        };
        const key = categoryKeyMap[categoryIdOrName];
        if (!key) return categoryIdOrName;
        const translated = t(key);
        return translated === key ? categoryIdOrName : translated;
    };

    const translateTemplateName = (template: UnifiedTemplate) => {
        if (template.source === 'user') return template.name;
        const translated = t(template.name);
        return translated === template.name ? template.name : translated;
    };

    // 当 open 变为 true 时，如果有 initialCategory，则选中该分类
    useEffect(() => {
        if (open && initialCategory) {
            setSelectedCategory(initialCategory);
        }
    }, [open, initialCategory]);

    // 搜索模板
    const filteredTemplates = useMemo(() => {
        if (!searchQuery.trim()) {
            return null; // 返回 null 表示显示分类视图
        }

        const query = searchQuery.toLowerCase();
        const results: UnifiedTemplate[] = [];

        categories.forEach(cat => {
            cat.templates.forEach(template => {
                if (
                    template.name.toLowerCase().includes(query) ||
                    template.content.toLowerCase().includes(query) ||
                    (template.keywords && template.keywords.some(k => k.toLowerCase().includes(query)))
                ) {
                    results.push(template);
                }
            });
        });

        return results;
    }, [categories, searchQuery]);

    // 处理使用模板
    const handleUse = (template: UnifiedTemplate) => {
        onInsert(template.content);
        addToRecent?.(template.id);
        onOpenChange(false);
    };

    // 渲染模板项
    const renderTemplateItem = (template: UnifiedTemplate, showCategory: boolean = false, compact: boolean = false) => {
        const IconComponent = template.icon ? getIconComponent(template.icon) : null;
        const isUserTemplate = template.source === 'user';

        if (compact) {
            return (
                <div
                    key={template.id}
                    className="group relative px-3 py-2 rounded-md border border-border bg-card hover:bg-accent/50 transition-colors cursor-pointer"
                    onClick={() => handleUse(template)}
                >
                    <div className="flex items-center gap-2">
                        {IconComponent && <IconComponent className="w-3.5 h-3.5 text-muted-foreground shrink-0" />}
                        <span className="font-medium text-xs truncate">{translateTemplateName(template)}</span>
                        {isUserTemplate ? (
                            <Badge variant="secondary" className="text-[11px] shrink-0 h-5 px-1">
                                <User className="w-3 h-3 mr-0.5" />
                                {t('userTemplates') || '我的'}
                            </Badge>
                        ) : (
                            <Badge variant="outline" className="text-[11px] shrink-0 h-5 px-1">
                                <BookOpen className="w-3 h-3 mr-0.5" />
                                {t('builtinTemplates') || '基础'}
                            </Badge>
                        )}
                        {showCategory && (
                            <span className="text-[11px] text-muted-foreground shrink-0">
                                {translateCategoryName(
                                    categories.find(c => c.id === template.category)?.name ||
                                        template.category,
                                )}
                            </span>
                        )}
                    </div>
                </div>
            );
        }

        return (
            <div
                key={template.id}
                className="group relative p-4 rounded-lg border border-border bg-card hover:bg-accent/50 transition-colors cursor-pointer w-full max-w-full overflow-hidden"
                onClick={() => handleUse(template)}
            >
                {/* 编辑删除按钮 - 右上角 hover 显示 */}
                {isUserTemplate && (
                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1 z-10">
                        {onEdit && (
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-7 w-7"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onEdit(template);
                                        }}
                                    >
                                        <Edit2 className="w-3.5 h-3.5" />
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent>{t('editTemplate') || '编辑'}</TooltipContent>
                            </Tooltip>
                        )}
                        {onDelete && (
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-7 w-7 text-destructive hover:text-destructive"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDelete(template.id);
                                        }}
                                    >
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent>{t('deleteTemplate') || '删除'}</TooltipContent>
                            </Tooltip>
                        )}
                    </div>
                )}
                
                {/* 模板标题和标签 */}
                <div className="flex items-center gap-2 mb-3 pr-16 min-w-0">
                    {IconComponent && <IconComponent className="w-4 h-4 text-muted-foreground shrink-0" />}
                    <span className="font-medium text-xs truncate">{translateTemplateName(template)}</span>
                    {isUserTemplate ? (
                        <Badge variant="secondary" className="text-[11px] shrink-0">
                            <User className="w-3 h-3 mr-1" />
                            {t('userTemplates') || '我的'}
                        </Badge>
                    ) : (
                        <Badge variant="outline" className="text-[11px] shrink-0">
                            <BookOpen className="w-3 h-3 mr-1" />
                            {t('builtinTemplates') || '基础'}
                        </Badge>
                    )}
                    {showCategory && (
                        <span className="text-[11px] text-muted-foreground shrink-0">
                            {translateCategoryName(
                                categories.find(c => c.id === template.category)?.name ||
                                    template.category,
                            )}
                        </span>
                    )}
                </div>
                
                {/* 完整内容 - 多行显示 */}
                <div className="text-[11px] text-muted-foreground whitespace-pre-wrap break-all bg-muted/30 rounded p-2 overflow-hidden">
                    {template.content}
                </div>
            </div>
        );
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-2xl h-[80vh] flex flex-col text-[13px]">
                <DialogHeader>
                    <DialogTitle className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                            <BookOpen className="w-5 h-5" />
                            {t('libraryTitle') || '模板库'}
                        </span>
                        {currentInput.trim().length > 0 && (
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                    onSaveNew();
                                    onOpenChange(false);
                                }}
                            >
                                <Plus className="w-4 h-4 mr-1" />
                                {t('saveCurrentAsTemplate') || '保存当前输入'}
                            </Button>
                        )}
                    </DialogTitle>
                </DialogHeader>

                {/* 搜索框 */}
                <div className="relative mt-4">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                        placeholder={t('searchTemplatesPlaceholder') || '搜索模板名称或内容...'}
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10 text-xs"
                    />
                </div>

                {/* 分类过滤（仅在非搜索模式下显示） */}
                {!searchQuery.trim() && (
                    <div className="flex gap-2 mt-4 flex-wrap text-xs">
                        <Button
                            variant={selectedCategory === null ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setSelectedCategory(null)}
                        >
                            {t('categoryAll') || '全部'}
                        </Button>
                        {categories.map((cat) => {
                            const IconComponent = getIconComponent(cat.icon);
                            return (
                                <Button
                                    key={cat.id}
                                    variant={selectedCategory === cat.id ? 'default' : 'outline'}
                                    size="sm"
                                    onClick={() => setSelectedCategory(cat.id === selectedCategory ? null : cat.id)}
                                >
                                    {IconComponent && <IconComponent className="w-4 h-4 mr-1" />}
                                    {translateCategoryName(cat.name)}
                                    <span className="ml-1 text-[11px] opacity-60">({cat.templates.length})</span>
                                </Button>
                            );
                        })}
                    </div>
                )}

                {/* 模板列表 */}
                <ScrollArea className="flex-1 mt-4 min-h-0 w-full">
                    {/* 最近使用 - 紧凑模式 */}
                    {!searchQuery.trim() && recentTemplates.length > 0 && (
                        <div className="mb-4 w-full min-w-0">
                            <h3 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-2">
                                <Clock className="w-4 h-4" />
                                {t('recentTemplates') || '最近使用'}
                            </h3>
                            <div className="flex flex-wrap gap-2">
                                {recentTemplates.slice(0, 3).map(template => renderTemplateItem(template, true, true))}
                            </div>
                            <Separator className="mt-3" />
                        </div>
                    )}

                    {/* 搜索结果 */}
                    {searchQuery.trim() ? (
                        <div className="w-full min-w-0">
                            <h3 className="text-xs font-semibold text-muted-foreground mb-3">
                                {t('templateSearchResults') || '搜索结果'}
                                {filteredTemplates && (
                                    <span className="ml-1 text-[11px] opacity-60">({filteredTemplates.length})</span>
                                )}
                            </h3>
                            {filteredTemplates && filteredTemplates.length > 0 ? (
                                <div className="space-y-2 w-full min-w-0">
                                    {filteredTemplates.map(template => renderTemplateItem(template, true))}
                                </div>
                            ) : (
                                <div className="text-center py-8 text-muted-foreground">
                                    {t('emptyTemplateSearch') || '没有找到匹配的模板'}
                                </div>
                            )}
                        </div>
                    ) : (
                        /* 分类列表 */
                        <div className="space-y-6 w-full">
                            {categories
                                .filter(cat => !selectedCategory || cat.id === selectedCategory)
                                .map((cat) => {
                                    const IconComponent = getIconComponent(cat.icon);
                                    const filteredTemplates = cat.templates;

                                    if (filteredTemplates.length === 0) return null;

                                    return (
                                        <div key={cat.id} className="w-full min-w-0">
                                            <h3 className="text-xs font-semibold text-muted-foreground mb-3 flex items-center gap-2">
                                                {IconComponent && <IconComponent className="w-4 h-4" />}
                                                {translateCategoryName(cat.name)}
                                                <span className="text-[11px] opacity-60">({filteredTemplates.length})</span>
                                            </h3>
                                            <div className="space-y-2 w-full min-w-0">
                                                {filteredTemplates.map(template => renderTemplateItem(template))}
                                            </div>
                                        </div>
                                    );
                                })}
                        </div>
                    )}
                </ScrollArea>

                {/* 底部提示 */}
                <div className="mt-4 pt-4 border-t text-[11px] text-muted-foreground text-center">
                    {t('clickTemplateToApply') || '点击模板直接应用到输入框'}
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default TemplateLibraryDialog;
