import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import {
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
} from '@/components/ui/tabs.tsx';
import { Dialog, DialogContent } from '@/components/ui/dialog.tsx';
import { useI18n } from '@/context/I18Context.tsx';
import { useTheme } from '@/context/ThemeContext.tsx';
import { settingsMenuItems } from '../config';

interface SettingsProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const Settings = ({ open, onOpenChange }: SettingsProps) => {
    const { t } = useTranslation();
    const { changeLanguage, currentLanguage } = useI18n();
    const { theme, setTheme } = useTheme();
    const [selectedLanguage, setSelectedLanguage] = useState(currentLanguage);

    // Update selected language when current language changes
    useEffect(() => {
        setSelectedLanguage(currentLanguage);
    }, [currentLanguage]);

    const handleLanguageChange = () => {
        changeLanguage();
    };

    // 渲染语言设置内容
    const renderLanguageContent = (item: typeof settingsMenuItems[0]) => {
        const Icon = item.icon;
        return (
            <div className="flex flex-col gap-6">
                <div>
                    <div className="flex items-center gap-2">
                        <Icon className="h-4 w-4" />
                        <div>
                            <h3 className="text-sm">
                                {t(item.labelKey)}
                            </h3>
                        </div>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                        {t(item.descriptionKey)}
                    </div>
                </div>
                <div className="flex flex-col bg-muted/50 rounded-lg p-4">
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-muted-foreground">
                            {t('settings.language-settings')}
                        </span>
                        <ToggleGroup
                            type="single"
                            value={selectedLanguage}
                            onValueChange={(value) => {
                                if (value && value !== currentLanguage) {
                                    setSelectedLanguage(value);
                                    handleLanguageChange();
                                }
                            }}
                            variant="outline"
                            size="sm"
                        >
                            <ToggleGroupItem
                                value="zh"
                                className="text-xs px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
                            >
                                中文
                            </ToggleGroupItem>
                            <ToggleGroupItem
                                value="en"
                                className="text-xs px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
                            >
                                English
                            </ToggleGroupItem>
                        </ToggleGroup>
                    </div>
                </div>
            </div>
        );
    };

    // 渲染外观设置内容
    const renderAppearanceContent = (item: typeof settingsMenuItems[0]) => {
        const Icon = item.icon;
        return (
            <div className="flex flex-col gap-6">
                <div>
                    <div className="flex items-center gap-2">
                        <Icon className="h-4 w-4" />
                        <div>
                            <h3 className="text-sm">
                                {t(item.labelKey)}
                            </h3>
                        </div>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                        {t(item.descriptionKey)}
                    </div>
                </div>
                <div className="flex flex-col bg-muted/50 rounded-lg p-4">
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-muted-foreground">
                            {t('settings.theme-settings')}
                        </span>
                        <ToggleGroup
                            type="single"
                            value={theme}
                            onValueChange={(value) => {
                                if (value) {
                                    setTheme(value as 'light' | 'dark');
                                }
                            }}
                            variant="outline"
                            size="sm"
                        >
                            <ToggleGroupItem
                                value="light"
                                className="text-xs px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
                            >
                                {t('settings.theme-light')}
                            </ToggleGroupItem>
                            <ToggleGroupItem
                                value="dark"
                                className="text-xs px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
                            >
                                {t('settings.theme-dark')}
                            </ToggleGroupItem>
                        </ToggleGroup>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[600px] p-0">
                <Tabs
                    defaultValue="language"
                    className="flex gap-6 items-start"
                >
                    <div className="h-[-webkit-fill-available] flex flex-col border-r border-border">
                        <h3 className="text-sm p-4 -mb-2 text-left ml-2">
                            {t('common.settings')}
                        </h3>
                        <TabsList className="flex flex-col h-auto bg-transparent p-2 w-[200px]">
                            {settingsMenuItems.map((item) => {
                                const Icon = item.icon;
                                return (
                                    <TabsTrigger
                                        key={item.value}
                                        value={item.value}
                                        className="w-full justify-start data-[state=active]:bg-muted gap-2 px-2 py-2 relative hover:bg-accent hover:text-accent-foreground"
                                    >
                                        <Icon className="size-4" />
                                        <span>{t(item.labelKey)}</span>
                                    </TabsTrigger>
                                );
                            })}
                        </TabsList>
                    </div>
                    <div className="flex-1 min-h-[300px]">
                        {settingsMenuItems.map((item) => {
                            return (
                                <TabsContent
                                    key={item.value}
                                    value={item.value}
                                    className="mt-4 pr-6"
                                >
                                    {item.value === 'language' && renderLanguageContent(item)}
                                    {item.value === 'appearance' && renderAppearanceContent(item)}
                                </TabsContent>
                            );
                        })}
                    </div>
                </Tabs>
            </DialogContent>
        </Dialog>
    );
};

export default memo(Settings);
