import { GlobeIcon, LucideIcon, Palette } from 'lucide-react';

export interface SettingsMenuItem {
    value: string;
    labelKey: string;
    descriptionKey: string;
    icon: LucideIcon;
}

/**
 * Settings menu items configuration for the settings dialog
 * Each item contains the tab value, i18n keys for label and description, and the icon component
 */
export const settingsMenuItems: SettingsMenuItem[] = [
    {
        value: 'language',
        labelKey: 'settings.language',
        descriptionKey: 'settings.language-description',
        icon: GlobeIcon,
    },
    {
        value: 'appearance',
        labelKey: 'settings.appearance',
        descriptionKey: 'settings.appearance-description',
        icon: Palette,
    },
];

/**
 * Sidebar menu item interface
 * Represents a single menu item in the sidebar navigation
 */
export interface SidebarSubItem {
    title: string;
    icon: LucideIcon | ComponentType<SVGProps<SVGSVGElement>>;
    url: string;
}

/**
 * Sidebar group interface
 * Represents a group of related menu items with a title
 */
export interface SidebarGroup {
    title: string;
    items: SidebarSubItem[];
}

/**
 * Sidebar navigation items configuration
 * Defines the main navigation structure for the application sidebar
 * Note: titleKey is the i18n translation key that will be resolved at runtime using t(titleKey)
 */
export const getSidebarItems = (_t: (key: string) => string): SidebarGroup[] => [
    // Navigation items removed - app auto-navigates to latest project
];
