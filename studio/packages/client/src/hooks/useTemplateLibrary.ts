import { useState, useEffect, useCallback } from 'react';
import { BookOpen, Clock, Dumbbell, BarChart3, Cpu, Database, Zap, Rocket, Calendar, Layers, GitCompare, Activity, CheckCircle, Play, Download, Settings, Filter, FileText, List, Square, LucideIcon } from 'lucide-react';

// 基础模板数据结构（来自 templates.json）
export interface BuiltinTemplate {
    id: string;
    title: string;
    content: string;
    description: string;
    icon: string;
    keywords: string[];
}

export interface BuiltinCategory {
    id: string;
    name: string;
    icon: string;
    templates: BuiltinTemplate[];
}

// 用户模板数据结构
export interface UserTemplate {
    id: string;
    name: string;
    content: string;
    category: string;
    createdAt: number;
    updatedAt: number;
    usageCount: number;
}

// 统一模板数据结构
export interface UnifiedTemplate {
    id: string;
    name: string;
    content: string;
    category: string;
    source: 'builtin' | 'user';
    isEditable: boolean;
    isDeletable: boolean;
    icon?: string;
    description?: string;
    keywords?: string[];
    createdAt?: number;
    updatedAt?: number;
    usageCount?: number;
}

// 统一分类数据结构
export interface UnifiedCategory {
    id: string;
    name: string;
    icon: string;
    templates: UnifiedTemplate[];
}

const USER_TEMPLATES_KEY = 'medflow-templates-v1';
const RECENT_TEMPLATES_KEY = 'medflow-templates-recent';
const MAX_RECENT = 3;
const RECENT_TEMPLATE_ID_MIGRATIONS: Record<string, string> = {
    'builtin-dual-model-evaluation': 'builtin-dual-model-assessment',
    'builtin-single-model-evaluation': 'builtin-single-model-assessment',
    'builtin-checkpoint-evaluation': 'builtin-checkpoint-assessment',
};
const ADMIN_ONLY_BUILTIN_TEMPLATE_IDS = new Set([
    'mgmt-system-query',
    'mgmt-gpu-status',
]);
const PRESET_CATEGORY_IDS = ['beginner', 'training', 'assessment', 'evaluation', 'inference', 'data'];

// 预设分类（用于新建模板时的分类选择）
export const PRESET_CATEGORIES = [
    { id: 'beginner', name: 'Beginner', icon: 'BookOpen' },
    { id: 'training', name: '训练', icon: 'Dumbbell' },
    { id: 'assessment', name: '医疗自建评估', icon: 'BarChart3' },
    { id: 'evaluation', name: '评测', icon: 'CheckCircle' },
    { id: 'inference', name: '推理', icon: 'Cpu' },
    { id: 'data', name: '数据', icon: 'Database' },
];

// Icon 映射
const iconMap: Record<string, LucideIcon> = {
    BookOpen,
    Clock,
    Dumbbell,
    BarChart3,
    Cpu,
    Database,
    Zap,
    Rocket,
    Calendar,
    Layers,
    GitCompare,
    Activity,
    CheckCircle,
    Play,
    Download,
    Settings,
    Filter,
    FileText,
    List,
    Square,
};

// 生成 UUID
const generateId = () => {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15);
};

const getCustomCategoryNames = (templates: UserTemplate[]) => {
    return [...new Set(templates.map(t => t.category))]
        .filter(c => ![...PRESET_CATEGORY_IDS, 'work', 'daily'].includes(c));
};

interface UseTemplateLibraryOptions {
    isAdmin?: boolean;
}

const filterBuiltinTemplatesByRole = (
    templates: BuiltinTemplate[],
    isAdmin: boolean,
) => {
    if (isAdmin) return templates;
    return templates.filter((template) => !ADMIN_ONLY_BUILTIN_TEMPLATE_IDS.has(template.id));
};

export const useTemplateLibrary = (options: UseTemplateLibraryOptions = {}) => {
    const [userTemplates, setUserTemplates] = useState<UserTemplate[]>([]);
    const [builtinCategories, setBuiltinCategories] = useState<BuiltinCategory[]>([]);
    const [recentIds, setRecentIds] = useState<string[]>([]);
    const [isLoaded, setIsLoaded] = useState(false);
    const [customCategories, setCustomCategories] = useState<string[]>([]);
    const isAdmin = options.isAdmin === true;

    // 加载基础模板
    useEffect(() => {
        const loadBuiltinTemplates = async () => {
            try {
                const response = await fetch('/src/components/chat/AsChat/templates.json');
                const data = await response.json();
                setBuiltinCategories(data.categories || []);
            } catch (error) {
                console.error('Failed to load builtin templates:', error);
                setBuiltinCategories([]);
            }
        };

        loadBuiltinTemplates();
    }, []);

    // 从 localStorage 加载用户模板
    useEffect(() => {
        const loadUserTemplates = () => {
            try {
                const saved = localStorage.getItem(USER_TEMPLATES_KEY);
                if (saved) {
                    const parsed = JSON.parse(saved);
                    if (parsed.version === 1 && parsed.templates) {
                        // 过滤掉 work 和 daily 分类的模板
                        const filteredTemplates = parsed.templates.filter(
                            (t: UserTemplate) => !['work', 'daily'].includes(t.category)
                        );
                        
                        // 如果有被过滤掉的模板，更新 localStorage
                        if (filteredTemplates.length !== parsed.templates.length) {
                            localStorage.setItem(USER_TEMPLATES_KEY, JSON.stringify({
                                version: 1,
                                templates: filteredTemplates,
                            }));
                        }
                        
                        setUserTemplates(filteredTemplates);
                        
                        // 提取自定义分类（排除预设分类和 work/daily）
                        setCustomCategories(getCustomCategoryNames(filteredTemplates));
                    }
                }
            } catch (error) {
                console.error('Failed to load user templates:', error);
            }
        };

        loadUserTemplates();
    }, []);

    // 加载最近使用记录
    useEffect(() => {
        const loadRecent = () => {
            try {
                const saved = localStorage.getItem(RECENT_TEMPLATES_KEY);
                if (saved) {
                    const parsedIds: string[] = JSON.parse(saved);
                    const migratedIds = parsedIds.map(id => RECENT_TEMPLATE_ID_MIGRATIONS[id] || id);
                    if (migratedIds.some((id, index) => id !== parsedIds[index])) {
                        localStorage.setItem(RECENT_TEMPLATES_KEY, JSON.stringify(migratedIds));
                    }
                    setRecentIds(migratedIds);
                }
            } catch (error) {
                console.error('Failed to load recent templates:', error);
            }
            setIsLoaded(true);
        };

        loadRecent();
    }, []);

    // 保存用户模板到 localStorage
    const saveUserTemplates = useCallback((templates: UserTemplate[]) => {
        try {
            localStorage.setItem(USER_TEMPLATES_KEY, JSON.stringify({
                version: 1,
                templates,
            }));
        } catch (error) {
            console.error('Failed to save user templates:', error);
        }
    }, []);

    // 保存最近使用记录
    const saveRecentIds = useCallback((ids: string[]) => {
        try {
            localStorage.setItem(RECENT_TEMPLATES_KEY, JSON.stringify(ids));
        } catch (error) {
            console.error('Failed to save recent templates:', error);
        }
    }, []);

    // 合并基础模板和用户模板为统一格式
    const getUnifiedCategories = useCallback((): UnifiedCategory[] => {
        const categories: UnifiedCategory[] = [];

        // 添加基础分类
        builtinCategories.forEach(builtinCat => {
            const templates: UnifiedTemplate[] = filterBuiltinTemplatesByRole(
                builtinCat.templates,
                isAdmin,
            ).map(t => ({
                id: `builtin-${t.id}`,
                name: t.title,
                content: t.content,
                category: builtinCat.id,
                source: 'builtin' as const,
                isEditable: false,
                isDeletable: false,
                icon: t.icon,
                description: t.description,
                keywords: t.keywords,
            }));

            if (templates.length === 0) {
                return;
            }

            categories.push({
                id: builtinCat.id,
                name: builtinCat.name,
                icon: builtinCat.icon,
                templates,
            });
        });

        // 添加用户模板
        userTemplates.forEach(userTemplate => {
            const existingCat = categories.find(c => c.id === userTemplate.category);
            const userTemplateUnified: UnifiedTemplate = {
                id: userTemplate.id,
                name: userTemplate.name,
                content: userTemplate.content,
                category: userTemplate.category,
                source: 'user' as const,
                isEditable: true,
                isDeletable: true,
                createdAt: userTemplate.createdAt,
                updatedAt: userTemplate.updatedAt,
                usageCount: userTemplate.usageCount,
            };

            if (existingCat) {
                existingCat.templates.push(userTemplateUnified);
            } else {
                // 创建新的自定义分类
                categories.push({
                    id: userTemplate.category,
                    name: userTemplate.category,
                    icon: 'BookOpen',
                    templates: [userTemplateUnified],
                });
            }
        });

        return categories;
    }, [builtinCategories, isAdmin, userTemplates]);

    // 获取最近使用的模板
    const getRecentTemplates = useCallback((): UnifiedTemplate[] => {
        const allTemplates: UnifiedTemplate[] = [];
        
        // 收集所有模板
        builtinCategories.forEach(cat => {
            filterBuiltinTemplatesByRole(cat.templates, isAdmin).forEach(t => {
                allTemplates.push({
                    id: `builtin-${t.id}`,
                    name: t.title,
                    content: t.content,
                    category: cat.id,
                    source: 'builtin',
                    isEditable: false,
                    isDeletable: false,
                    icon: t.icon,
                });
            });
        });

        userTemplates.forEach(t => {
            allTemplates.push({
                id: t.id,
                name: t.name,
                content: t.content,
                category: t.category,
                source: 'user',
                isEditable: true,
                isDeletable: true,
                createdAt: t.createdAt,
                updatedAt: t.updatedAt,
                usageCount: t.usageCount,
            });
        });

        // 按最近使用顺序返回
        return recentIds
            .map(id => allTemplates.find(t => t.id === id))
            .filter((t): t is UnifiedTemplate => t !== undefined);
    }, [builtinCategories, isAdmin, userTemplates, recentIds]);

    // 添加模板到最近使用
    const addToRecent = useCallback((templateId: string) => {
        setRecentIds(prev => {
            const filtered = prev.filter(id => id !== templateId);
            const updated = [templateId, ...filtered].slice(0, MAX_RECENT);
            saveRecentIds(updated);
            return updated;
        });

        setUserTemplates(prev => {
            const updated = prev.map(t =>
                t.id === templateId
                    ? { ...t, usageCount: t.usageCount + 1, updatedAt: Date.now() }
                    : t
            );
            if (updated === prev || updated.every((t, index) => t === prev[index])) {
                return prev;
            }
            saveUserTemplates(updated);
            return updated;
        });
    }, [saveRecentIds, saveUserTemplates]);

    // 添加自定义分类
    const addCustomCategory = useCallback((name: string) => {
        const category = name.trim();
        if (!category || PRESET_CATEGORY_IDS.includes(category) || ['work', 'daily'].includes(category)) {
            return category;
        }

        setCustomCategories(prev => (
            prev.some(c => c.toLowerCase() === category.toLowerCase())
                ? prev
                : [...prev, category]
        ));
        return category;
    }, []);

    // 添加用户模板
    const addTemplate = useCallback((name: string, content: string, category: string) => {
        const normalizedCategory = category.trim();
        const newTemplate: UserTemplate = {
            id: generateId(),
            name: name.trim(),
            content,
            category: normalizedCategory,
            createdAt: Date.now(),
            updatedAt: Date.now(),
            usageCount: 0,
        };

        const updated = [newTemplate, ...userTemplates];
        setUserTemplates(updated);
        saveUserTemplates(updated);

        // 如果是新分类，添加到自定义分类列表
        if (!customCategories.includes(normalizedCategory) && 
            !PRESET_CATEGORY_IDS.includes(normalizedCategory)) {
            setCustomCategories(prev => [...prev, normalizedCategory]);
        }

        return newTemplate;
    }, [userTemplates, customCategories, saveUserTemplates]);

    // 更新用户模板
    const updateTemplate = useCallback((id: string, updates: Partial<Omit<UserTemplate, 'id' | 'createdAt'>>) => {
        const updated = userTemplates.map(t => 
            t.id === id 
                ? { ...t, ...updates, updatedAt: Date.now() } 
                : t
        );
        setUserTemplates(updated);
        saveUserTemplates(updated);
        setCustomCategories(getCustomCategoryNames(updated));
    }, [userTemplates, saveUserTemplates]);

    // 删除用户模板
    const deleteTemplate = useCallback((id: string) => {
        const updated = userTemplates.filter(t => t.id !== id);
        setUserTemplates(updated);
        saveUserTemplates(updated);
        setCustomCategories(getCustomCategoryNames(updated));
        
        // 从最近使用中也删除
        setRecentIds(prev => {
            const updated = prev.filter(recentId => recentId !== id);
            saveRecentIds(updated);
            return updated;
        });
    }, [userTemplates, saveUserTemplates, saveRecentIds]);

    // 获取单个模板
    const getTemplate = useCallback((id: string): UnifiedTemplate | undefined => {
        // 先查找用户模板
        const userTemplate = userTemplates.find(t => t.id === id);
        if (userTemplate) {
            return {
                id: userTemplate.id,
                name: userTemplate.name,
                content: userTemplate.content,
                category: userTemplate.category,
                source: 'user',
                isEditable: true,
                isDeletable: true,
                createdAt: userTemplate.createdAt,
                updatedAt: userTemplate.updatedAt,
                usageCount: userTemplate.usageCount,
            };
        }

        // 再查找基础模板
        for (const cat of builtinCategories) {
            const builtinTemplate = filterBuiltinTemplatesByRole(
                cat.templates,
                isAdmin,
            ).find(t => `builtin-${t.id}` === id);
            if (builtinTemplate) {
                return {
                    id: `builtin-${builtinTemplate.id}`,
                    name: builtinTemplate.title,
                    content: builtinTemplate.content,
                    category: cat.id,
                    source: 'builtin',
                    isEditable: false,
                    isDeletable: false,
                    icon: builtinTemplate.icon,
                    description: builtinTemplate.description,
                    keywords: builtinTemplate.keywords,
                };
            }
        }

        return undefined;
    }, [userTemplates, builtinCategories, isAdmin]);

    // 搜索模板
    const searchTemplates = useCallback((query: string): UnifiedTemplate[] => {
        const lowerQuery = query.toLowerCase();
        const allTemplates: UnifiedTemplate[] = [];

        // 收集所有模板
        getUnifiedCategories().forEach(cat => {
            allTemplates.push(...cat.templates);
        });

        return allTemplates.filter(t => 
            t.name.toLowerCase().includes(lowerQuery) || 
            t.content.toLowerCase().includes(lowerQuery) ||
            (t.keywords && t.keywords.some(k => k.toLowerCase().includes(lowerQuery)))
        );
    }, [getUnifiedCategories]);

    // 获取 Icon 组件
    const getIconComponent = useCallback((iconName: string): LucideIcon | undefined => {
        return iconMap[iconName];
    }, []);

    return {
        isLoaded,
        userTemplates,
        builtinCategories,
        recentIds,
        customCategories,
        getUnifiedCategories,
        getRecentTemplates,
        addToRecent,
        addCustomCategory,
        addTemplate,
        updateTemplate,
        deleteTemplate,
        getTemplate,
        searchTemplates,
        getIconComponent,
    };
};

export default useTemplateLibrary;
