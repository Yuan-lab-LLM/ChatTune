import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Plus, Save, ChevronDown } from 'lucide-react';
import { PRESET_CATEGORIES, UserTemplate } from '@/hooks/useTemplateLibrary';

interface SaveTemplateDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    content: string;
    onSave: (name: string, content: string, category: string) => void;
    customCategories: string[];
    onAddCategory: (name: string) => void;
    mode?: 'create' | 'edit';
    initialTemplate?: UserTemplate | null;
    existingTemplates?: UserTemplate[];
}

const SaveTemplateDialog = ({
    open,
    onOpenChange,
    content,
    onSave,
    customCategories,
    onAddCategory,
    mode = 'create',
    initialTemplate = null,
    existingTemplates = [],
}: SaveTemplateDialogProps) => {
    const { t } = useTranslation();
    const [name, setName] = useState('');
    const [category, setCategory] = useState('training');
    const [newCategoryName, setNewCategoryName] = useState('');
    const [isAddingCategory, setIsAddingCategory] = useState(false);
    const [error, setError] = useState('');

    // 重置表单
    useEffect(() => {
        if (open) {
            setName(initialTemplate?.name || '');
            setCategory(initialTemplate?.category || 'training');
            setNewCategoryName('');
            setIsAddingCategory(false);
            setError('');
        }
    }, [open, initialTemplate]);

    // 获取所有分类选项
    const allCategories = [
        ...PRESET_CATEGORIES.map(c => ({ 
            value: c.id, 
            label: t(`category${c.id.charAt(0).toUpperCase() + c.id.slice(1)}`) || c.name
        })),
        ...customCategories.filter(c => !['work', 'daily'].includes(c)).map(c => ({ value: c, label: c })),
    ];

    const handleSave = () => {
        const trimmedName = name.trim();
        const trimmedContent = content.trim();

        // 验证
        if (!trimmedName) {
            setError(t('templateNameRequired') || '请输入模板名称');
            return;
        }

        if (trimmedName.length > 50) {
            setError(t('templateNameTooLong') || '模板名称不能超过50个字符');
            return;
        }

        if (!trimmedContent) {
            setError(t('templateContentRequired') || '模板内容不能为空');
            return;
        }

        const duplicated = existingTemplates.some(template =>
            template.id !== initialTemplate?.id &&
            template.category === category &&
            template.name.trim().toLowerCase() === trimmedName.toLowerCase()
        );

        if (duplicated) {
            setError(t('templateNameDuplicated') || '同一分类下已存在同名模板');
            return;
        }

        onSave(trimmedName, content, category);
        onOpenChange(false);
    };

    const handleAddCategory = () => {
        const categoryName = newCategoryName.trim();
        if (!categoryName) return;

        if (categoryName.length > 30) {
            setError(t('templateCategoryTooLong') || '分类名称不能超过30个字符');
            return;
        }

        const duplicateCategory = allCategories.some(
            cat => cat.value.toLowerCase() === categoryName.toLowerCase() ||
                cat.label.toLowerCase() === categoryName.toLowerCase(),
        );
        if (duplicateCategory) {
            setError(t('templateCategoryDuplicated') || '该分类已存在');
            return;
        }

        onAddCategory(categoryName);
        setCategory(categoryName);
        setNewCategoryName('');
        setIsAddingCategory(false);
        setError('');
    };

    // 截断内容预览
    const truncateContent = (content: string, maxLength: number = 300) => {
        if (content.length <= maxLength) return content;
        return content.substring(0, maxLength) + '...';
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-lg">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Save className="w-5 h-5" />
                        {mode === 'edit'
                            ? (t('editTemplate') || '编辑模板')
                            : (t('saveTemplate') || '保存为模板')}
                    </DialogTitle>
                </DialogHeader>

                <div className="space-y-4 mt-4">
                    {/* 模板名称 */}
                    <div className="space-y-2">
                        <Label htmlFor="template-name">
                            {t('templateName') || '模板名称'}
                            <span className="text-red-500 ml-1">*</span>
                        </Label>
                        <Input
                            id="template-name"
                            placeholder={t('templateNamePlaceholder') || '例如：代码审查模板'}
                            value={name}
                            onChange={(e) => {
                                setName(e.target.value);
                                setError('');
                            }}
                            className={error ? 'border-red-500' : ''}
                        />
                        {error && (
                            <p className="text-sm text-red-500">{error}</p>
                        )}
                    </div>

                    {/* 分类选择 */}
                    <div className="space-y-2">
                        <Label>{t('templateCategory') || '分类'}</Label>
                        {!isAddingCategory ? (
                            <div className="flex gap-2">
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button variant="outline" className="flex-1 justify-between">
                                            {allCategories.find(c => c.value === category)?.label || category}
                                            <ChevronDown className="w-4 h-4 ml-2" />
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent className="w-[200px]">
                                        {allCategories.map((cat) => (
                                            <DropdownMenuItem 
                                                key={cat.value} 
                                                onClick={() => setCategory(cat.value)}
                                            >
                                                {cat.label}
                                            </DropdownMenuItem>
                                        ))}
                                    </DropdownMenuContent>
                                </DropdownMenu>
                                <Button
                                    variant="outline"
                                    size="icon"
                                    onClick={() => setIsAddingCategory(true)}
                                    title={t('newTemplateCategory') || '新建分类'}
                                >
                                    <Plus className="w-4 h-4" />
                                </Button>
                            </div>
                        ) : (
                            <div className="flex gap-2">
                                <Input
                                    placeholder={t('newTemplateCategoryPlaceholder') || '输入新分类名称'}
                                    value={newCategoryName}
                                    onChange={(e) => setNewCategoryName(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                            handleAddCategory();
                                        }
                                    }}
                                    autoFocus
                                />
                                <Button
                                    variant="default"
                                    size="sm"
                                    onClick={handleAddCategory}
                                    disabled={!newCategoryName.trim()}
                                >
                                    {t('action.add') || '添加'}
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => {
                                        setIsAddingCategory(false);
                                        setNewCategoryName('');
                                    }}
                                >
                                    {t('action.cancel') || '取消'}
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* 内容预览 */}
                    <div className="space-y-2">
                        <Label>{t('templatePreview') || '内容预览'}</Label>
                        <ScrollArea className="h-[200px] w-full rounded-md border p-4 bg-muted/30">
                            <pre className="text-sm whitespace-pre-wrap font-mono">
                                {truncateContent(content)}
                            </pre>
                        </ScrollArea>
                        <p className="text-xs text-muted-foreground">
                            {t('templateContentLength', { length: content.length }) ||
                                `内容长度：${content.length} 字符`}
                        </p>
                    </div>
                </div>

                <DialogFooter className="mt-6">
                    <Button variant="ghost" onClick={() => onOpenChange(false)}>
                        {t('action.cancel') || '取消'}
                    </Button>
                    <Button onClick={handleSave} disabled={!name.trim()}>
                        <Save className="w-4 h-4 mr-2" />
                        {mode === 'edit'
                            ? (t('updateTemplateButton') || '更新')
                            : (t('saveTemplateButton') || '保存')}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};

export default SaveTemplateDialog;
