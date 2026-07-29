import { ConfigProvider } from 'antd';
import { BrowserRouter } from 'react-router-dom';
import './App.css';
import { QueryClientProvider } from '@tanstack/react-query';
import { I18nProvider } from './context/I18Context.tsx';
import { trpc, queryClient, trpcClient } from './api/trpc';
import { MessageApiContextProvider } from './context/MessageApiContext.tsx';
import { NotificationContextProvider } from './context/NotificationContext.tsx';
import { SocketContextProvider } from './context/SocketContext.tsx';
import { ThemeContextProvider } from './context/ThemeContext.tsx';
import { RunsTabConfigProvider } from './context/RunsTabConfigContext.tsx';
import { FirstTimeGuideProvider } from './context/FirstTimeGuideContext.tsx';
import { AppTour } from './components/FirstTimeGuide/AppTour.tsx';
import { AuthProvider, useAuth } from './context/AuthContext.tsx';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import ForcePasswordChangePage from './pages/ForcePasswordChangePage';
import { useTranslation } from 'react-i18next';

const AuthenticatedApp = () => {
    const { t } = useTranslation();
    const { isAuthenticated, isChecking, user } = useAuth();

    if (isChecking) {
        return (
            <main className="auth-page">
                <section className="auth-panel auth-loading">
                    {t('auth.checking')}
                </section>
            </main>
        );
    }

    if (!isAuthenticated) {
        return <LoginPage />;
    }

    if (user?.mustChangePassword) {
        return <ForcePasswordChangePage />;
    }

    return (
        <SocketContextProvider>
            <BrowserRouter>
                <FirstTimeGuideProvider>
                    <HomePage />
                    <AppTour />
                </FirstTimeGuideProvider>
            </BrowserRouter>
        </SocketContextProvider>
    );
};

function App() {
    return (
        <ThemeContextProvider>
            <ConfigProvider
                theme={{
                    token: {
                        colorText: 'var(--foreground)',
                        colorTextSecondary: 'var(--muted-foreground)',
                        colorInfo: 'var(--primary)',
                        colorPrimary: 'var(--primary)',
                        colorPrimaryBorder: 'var(--border)',
                        colorPrimaryHover: 'var(--primary)',
                        fontFamily:
                            'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFon, Segoe UI, Roboto, Helvetica Neue, Arial, Noto Sans, sans-serif',
                        // For Button
                        colorBgSolidActive: 'var(--primary-700)',
                        colorBgSolidHover: 'var(--primary-800)',
                    },
                    components: {
                        Table: {
                            headerBg: 'var(--muted)',
                            headerColor: 'var(--muted-foreground)',
                            headerBorderRadius: 5,
                            colorBgContainer: 'var(--background)',
                            colorText: 'var(--foreground)',
                            colorTextHeading: 'var(--foreground)',
                            borderColor: 'var(--border)',
                        },
                        Tabs: {
                            titleFontSizeSM: 12,
                        },
                        Tree: {
                            nodeSelectedColor: 'var(--primary-foreground)',
                        },
                        Menu: {
                            itemSelectedColor: 'var(--sidebar-primary-foreground)',
                            itemSelectedBg: 'var(--sidebar-primary)',
                            itemColor: 'var(--sidebar-foreground)',
                            groupTitleColor: 'var(--muted-foreground)',
                            groupTitleFontSize: 12,
                            itemBg: 'transparent',
                        },
                        Layout: {
                            headerBg: 'var(--background)',
                            siderBg: 'var(--sidebar)',
                            bodyBg: 'var(--background)',
                        },
                        Collapse: {
                            headerBg: 'var(--primary-800)',
                            colorTextHeading: 'var(--primary-50)',
                            contentPadding: '0 !important',
                        },
                        Input: {
                            activeShadow: 'none',
                        },
                        InputNumber: {
                            activeShadow: 'none',
                        },
                        Select: {
                            activeOutlineColor: 'none',
                        },
                        Tooltip: {
                            colorBgSpotlight: 'var(--popover)',
                            colorText: 'var(--popover-foreground)',
                            colorTextLightSolid: 'var(--popover-foreground)',
                        },
                        Button: {
                            // 主按钮颜色配置 - iOS 蓝
                            colorPrimary: 'var(--primary)',
                            colorPrimaryHover: 'var(--primary-600)',
                            colorPrimaryActive: 'var(--primary-700)',
                            colorTextLightSolid: 'var(--primary-foreground)',
                            
                            primaryShadow: 'none',
                            contentFontSize: 12,
                            contentFontSizeLG: 13,
                            contentFontSizeSM: 11,
                            borderColorDisabled: 'var(--border)',
                            solidTextColor: 'var(--primary-foreground)',

                            defaultBg: 'var(--secondary)',
                            defaultColor: 'var(--secondary-foreground)',

                            defaultHoverBg: 'var(--secondary-hover)',
                            defaultHoverColor: 'var(--secondary-hover-foreground)',

                            defaultActiveBg: 'var(--secondary-active)',
                            defaultActiveColor:
                                'var(--secondary-active-foreground)',

                            defaultBorderColor: 'var(--border)',
                            defaultHoverBorderColor: 'var(--border)',
                            defaultActiveBorderColor: 'var(--border)',

                            fontWeight: 500,

                            // with icon and text
                            paddingInline: 13,
                        },
                        Statistic: {
                            contentFontSize: 14,
                            titleFontSize: 12,
                        },
                        Pagination: {
                            colorBgContainer: 'var(--background)',
                            colorText: 'var(--foreground)',
                            colorPrimary: 'var(--primary)',
                            itemActiveBg: 'var(--primary)',
                            itemActiveColor: 'var(--primary-foreground)',
                            itemHoverBg: 'var(--muted)',
                            itemHoverColor: 'var(--foreground)',
                        },
                        Dropdown: {
                            colorBgElevated: 'var(--popover)',
                            colorText: 'var(--popover-foreground)',
                            controlItemBgHover: 'var(--muted)',
                            controlItemBgActive: 'var(--primary)',
                            controlItemTextActive: 'var(--primary-foreground)',
                        },
                        Card: {
                            colorBgContainer: 'var(--card)',
                            colorBorderSecondary: 'var(--border)',
                            colorText: 'var(--card-foreground)',
                        },
                        Modal: {
                            colorBgElevated: 'var(--popover)',
                            colorText: 'var(--popover-foreground)',
                            colorTextHeading: 'var(--popover-foreground)',
                            colorIcon: 'var(--muted-foreground)',
                            colorIconHover: 'var(--foreground)',
                        },
                    },
                }}
            >
                <MessageApiContextProvider>
                    <NotificationContextProvider>
                        <trpc.Provider
                            client={trpcClient}
                            queryClient={queryClient}
                        >
                            <QueryClientProvider client={queryClient}>
                                <I18nProvider>
                                    <RunsTabConfigProvider>
                                        <AuthProvider>
                                            <AuthenticatedApp />
                                        </AuthProvider>
                                    </RunsTabConfigProvider>
                                </I18nProvider>
                            </QueryClientProvider>
                        </trpc.Provider>
                    </NotificationContextProvider>
                </MessageApiContextProvider>
            </ConfigProvider>
        </ThemeContextProvider>
    );
}

export default App;

