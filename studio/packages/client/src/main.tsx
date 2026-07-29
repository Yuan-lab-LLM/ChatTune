import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';
import './i18n/config';

// 主题初始化：防止页面闪烁
const initializeTheme = () => {
    const THEME_STORAGE_KEY = 'theme-mode';
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) as 'light' | 'dark' | 'system' | null;
    const theme = savedTheme || 'light';
    
    let resolvedTheme: 'light' | 'dark';
    if (theme === 'system') {
        resolvedTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    } else {
        resolvedTheme = theme;
    }
    
    document.documentElement.classList.add(resolvedTheme);
};

initializeTheme();

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <App />
    </StrictMode>,
);
