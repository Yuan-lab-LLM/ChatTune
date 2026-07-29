import { createContext, useCallback, useContext, useEffect, useState } from 'react';

import { ThemeColors } from '../theme/color.ts';
import { updateCSSVariables } from '../theme/utils.ts';

const defaultTheme: ThemeColors = {
    primary: '#',
};

type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeContextType {
    colors: ThemeColors;
    setThemeColor: (color: string) => void;
    theme: ThemeMode;
    setTheme: (theme: ThemeMode) => void;
    resolvedTheme: 'light' | 'dark';
}

const ThemeContext = createContext<ThemeContextType>({
    colors: defaultTheme,
    setThemeColor: () => {},
    theme: 'light',
    setTheme: () => {},
    resolvedTheme: 'light',
});

const THEME_STORAGE_KEY = 'theme-mode';

export function ThemeContextProvider({
    children,
}: {
    children: React.ReactNode;
}) {
    const [colors, setColors] = useState<ThemeColors>(defaultTheme);
    const [theme, setThemeState] = useState<ThemeMode>(() => {
        // 从 localStorage 读取保存的主题设置，默认亮色模式
        if (typeof window !== 'undefined') {
            const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
            return savedTheme || 'light';
        }
        return 'light';
    });
    const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light');

    // 设置主题并保存到 localStorage
    const setTheme = useCallback((newTheme: ThemeMode) => {
        setThemeState(newTheme);
        localStorage.setItem(THEME_STORAGE_KEY, newTheme);
    }, []);

    // 监听系统主题偏好变化
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        
        const updateResolvedTheme = () => {
            if (theme === 'system') {
                setResolvedTheme(mediaQuery.matches ? 'dark' : 'light');
            } else {
                setResolvedTheme(theme);
            }
        };

        updateResolvedTheme();

        if (theme === 'system') {
            mediaQuery.addEventListener('change', updateResolvedTheme);
        }

        return () => {
            mediaQuery.removeEventListener('change', updateResolvedTheme);
        };
    }, [theme]);

    // 应用主题到 document
    useEffect(() => {
        const root = window.document.documentElement;
        root.classList.remove('light', 'dark');
        root.classList.add(resolvedTheme);
    }, [resolvedTheme]);

    const setThemeColor = useCallback(
        (newPrimaryColor: string) => {
            const newColors = {
                ...colors,
                primary: newPrimaryColor,
            };
            setColors(newColors);
            updateCSSVariables(newColors);
        },
        [colors],
    );

    return (
        <ThemeContext.Provider value={{ colors, setThemeColor, theme, setTheme, resolvedTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export const useTheme = () => useContext(ThemeContext);
