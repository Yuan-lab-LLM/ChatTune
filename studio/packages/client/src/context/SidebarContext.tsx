import { useEffect, useState } from 'react';
import { createContext, ReactNode, useContext } from 'react';
import { useLocation } from 'react-router-dom';
import { trpc } from '@/api/trpc';
import { useTranslation } from 'react-i18next';
import { useMessageApi } from '@/context/MessageApiContext.tsx';
import { useAuth } from '@/context/AuthContext.tsx';

interface DatabaseInfoType {
    size: number;
    formattedSize: string;
    path: string;
}
export type RunPageSection =
    | 'runs'
    | 'overview'
    | 'datasets'
    | 'models'
    | 'evaluation';
interface StudioSidebarContextType {
    clearDataDialogOpen: boolean;
    latestVersion: string;
    currentVersion: string;
    databaseInfo?: DatabaseInfoType | null;
    showRunPageNavigation: boolean;
    runPageSection: RunPageSection;
    isRunPagePanelOpen: boolean;
    confirmClearData: () => void;
    setClearDataDialogOpen: (open: boolean) => void;
    setLatestVersion: (version: string) => void;
    setRunPagePanelOpen: (open: boolean) => void;
    setRunPageSection: (section: RunPageSection) => void;
}

const StudioSidebarContext = createContext<StudioSidebarContextType | null>(
    null,
);

export const StudioSidebarProvider = ({
    children,
}: {
    children: ReactNode;
}) => {
    const { t } = useTranslation();
    const { messageApi } = useMessageApi();
    const location = useLocation();
    const { isAdmin } = useAuth();

    const [clearDataDialogOpen, setClearDataDialogOpen] = useState(false);
    const [latestVersion, setLatestVersion] = useState<string>('');
    const defaultRunPageSection: RunPageSection = isAdmin ? 'runs' : 'overview';
    const [runPageSection, setRunPageSection] = useState<RunPageSection>(defaultRunPageSection);
    const [isRunPagePanelOpen, setRunPagePanelOpen] = useState(false);
    const { data: currentVersionData } = trpc.getCurrentVersion.useQuery();
    const { data: databaseInfo } = trpc.getDataInfo.useQuery(undefined, {
        enabled: isAdmin,
    });

    const showRunPageNavigation = /^\/projects\/[^/]+(?:\/.*)?$/.test(
        location.pathname,
    );

    useEffect(() => {
        if (!showRunPageNavigation) {
            setRunPageSection(defaultRunPageSection);
            setRunPagePanelOpen(false);
        }
    }, [defaultRunPageSection, showRunPageNavigation]);

    useEffect(() => {
        if (!isAdmin && runPageSection === 'runs') {
            setRunPageSection('overview');
        }
    }, [isAdmin, runPageSection]);

    const confirmClearData = () => {
        messageApi.info(t('message.settings.data-cleared'));
        setClearDataDialogOpen(false);
    };

    const value: StudioSidebarContextType = {
        clearDataDialogOpen,
        latestVersion,
        currentVersion: currentVersionData?.data?.version || '',
        databaseInfo: databaseInfo?.data || null,
        showRunPageNavigation,
        runPageSection,
        isRunPagePanelOpen,
        confirmClearData,
        setLatestVersion,
        setClearDataDialogOpen,
        setRunPagePanelOpen,
        setRunPageSection,
    };

    return (
        <StudioSidebarContext.Provider value={value}>
            {children}
        </StudioSidebarContext.Provider>
    );
};

export const useStudioSidebar = () => {
    const context = useContext(StudioSidebarContext);
    if (!context) {
        throw new Error(
            'useStudioSidebar must be used within a StudioSidebarProvider',
        );
    }
    return context;
};
