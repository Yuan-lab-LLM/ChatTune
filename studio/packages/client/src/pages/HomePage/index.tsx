import { useEffect } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar.tsx';

import { RouterPath } from '../RouterPath.ts';
import { checkForUpdates } from '@/utils/versionCheck.ts';
import { useNotification } from '@/context/NotificationContext.tsx';
import { StudioSidebarProvider } from '@/context/SidebarContext.tsx';
import { WandbProvider } from '@/context/WandbContext.tsx';
import { OverviewRoomContextProvider } from '@/context/OverviewRoomContext.tsx';
import StudioSidebar from '@/pages/HomePage/sidebar.tsx';
import LandingPage from '@/pages/LandingPage';
import RunPage from '@/pages/DashboardPage/RunPage';
import ProjectPage from '@/pages/DashboardPage/ProjectPage';
import { TourContextProvider } from '@/context/TourContext.tsx';
import { ProjectListRoomContextProvider } from '@/context/ProjectListRoomContext.tsx';
import { useAuth } from '@/context/AuthContext.tsx';
import UserLandingPage from '@/pages/LandingPage/UserLandingPage';

const HomePage = () => {
    const { t } = useTranslation();
    const { notificationApi } = useNotification();
    const { isAdmin } = useAuth();

    // Check for update
    useEffect(() => {
        const checkUpdate = async () => {
            const CHECK_INTERVAL = 5 * 24 * 60 * 60 * 1000; // 5 * 24小时
            const lastCheck = localStorage.getItem('lastUpdateCheck');
            const now = Date.now();

            if (!lastCheck || now - Number(lastCheck) > CHECK_INTERVAL) {
                const updateInfo = await checkForUpdates();
                if (updateInfo.hasUpdate) {
                    notificationApi.info({
                        message: t('notification.update-version-title'),
                        description: t(
                            'notification.update-version-description',
                            {
                                latestVersion: updateInfo.latestVersion,
                                currentVersion: updateInfo.currentVersion,
                            },
                        ),
                        placement: 'topRight',
                        duration: 5,
                    });
                }
                localStorage.setItem('lastUpdateCheck', String(now));
            }
        };
        checkUpdate();
    }, []);

    return (
        <WandbProvider>
            <SidebarProvider
                defaultOpen={true}
                style={
                    {
                        '--sidebar-width': '10rem',
                        '--sidebar-width-icon': '3.2rem',
                    } as React.CSSProperties
                }
            >
                <StudioSidebarProvider>
                    <StudioSidebar />
                    <SidebarInset>
                        <Routes>
                            <Route
                                path="/"
                                element={
                                    <OverviewRoomContextProvider>
                                        {isAdmin ? <LandingPage /> : <UserLandingPage />}
                                    </OverviewRoomContextProvider>
                                }
                            />
                            <Route path="/home" element={<Navigate to="/" replace />} />
                            <Route
                                path={RouterPath.PROJECTS}
                                element={
                                    <OverviewRoomContextProvider>
                                        {isAdmin ? <LandingPage /> : <UserLandingPage />}
                                    </OverviewRoomContextProvider>
                                }
                            />
                            <Route
                                path={`${RouterPath.PROJECTS}/list`}
                                element={
                                    isAdmin ? (
                                        <ProjectListRoomContextProvider>
                                            <ProjectPage />
                                        </ProjectListRoomContextProvider>
                                    ) : (
                                        <Navigate to="/" replace />
                                    )
                                }
                            />
                            <Route
                                path={`${RouterPath.PROJECTS}/:projectName/*`}
                                element={
                                    <TourContextProvider>
                                        <RunPage />
                                    </TourContextProvider>
                                }
                            />
                            <Route path="*" element={<Navigate to="/" replace />} />
                        </Routes>
                    </SidebarInset>
                </StudioSidebarProvider>
            </SidebarProvider>
        </WandbProvider>
    );
};

export default HomePage;
