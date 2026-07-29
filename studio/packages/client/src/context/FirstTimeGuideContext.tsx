import { createContext, ReactNode, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const GUIDE_STORAGE_KEY = 'firstTimeGuide_v1';

interface FirstTimeGuideContextType {
    showAutoPopup: boolean;
    showTour: boolean;
    showPostTourChoice: boolean;
    isWaitingForProjectLoad: boolean;
    startTour: (projectName?: string) => void;
    closeAutoPopup: () => void;
    completeTour: () => void;
    closeTourWithoutCompleting: () => void;
    closePostTourChoice: () => void;
}

const FirstTimeGuideContext = createContext<FirstTimeGuideContextType | null>(null);

export function FirstTimeGuideProvider({ children }: { children: ReactNode }) {
    const navigate = useNavigate();
    const location = useLocation();
    const [showAutoPopup, setShowAutoPopup] = useState(false);
    const [showTour, setShowTour] = useState(false);
    const [showPostTourChoice, setShowPostTourChoice] = useState(false);
    const [isWaitingForProjectLoad, setIsWaitingForProjectLoad] = useState(false);
    const pendingProjectNameRef = useRef<string | null>(null);
    const checkIntervalRef = useRef<number | null>(null);

    // 清理检查定时器
    const clearCheckInterval = useCallback(() => {
        if (checkIntervalRef.current) {
            clearInterval(checkIntervalRef.current);
            checkIntervalRef.current = null;
        }
    }, []);
    
    // 检查页面是否已加载 - 检查左侧一级菜单是否存在
    const checkProjectLoaded = useCallback(() => {
        const primaryNav = document.querySelector('[data-runpage-nav="true"]') as HTMLElement;
        const hasPrimaryNav = !!primaryNav && primaryNav.offsetHeight > 0;
        const runsNavButton = document.querySelector('[data-section="runs"]') as HTMLElement;
        const hasRunsNavButton = !!runsNavButton && runsNavButton.offsetHeight > 0;
        
        // 检查是否在正确的页面路径（/projects/{name}/runs 或 /projects/{name}）
        const isOnProjectPath = location.pathname.match(/\/projects\/[^/]+/);
        const isOnRunsPath = location.pathname.includes('/runs') || location.pathname.match(/\/projects\/[^/]+$/);
        
        
        return hasPrimaryNav && hasRunsNavButton && isOnRunsPath;
    }, [location.pathname]);
    
    const startTour = useCallback((projectName?: string) => {
        const targetProjectName = projectName || pendingProjectNameRef.current;
        pendingProjectNameRef.current = targetProjectName || null;
        
        
        if (targetProjectName) {
            const targetPath = `/projects/${targetProjectName}/runs`;
            const isAlreadyOnTarget = location.pathname === targetPath;
            
            if (!isAlreadyOnTarget) {
                // 需要导航到目标页面
                setIsWaitingForProjectLoad(true);
                setShowAutoPopup(false);
                navigate(targetPath);
            } else {
                // 已经在目标页面，检查数据是否已加载
                if (checkProjectLoaded()) {
                    // 已加载，直接开始引导
                    setShowTour(true);
                    setShowAutoPopup(false);
                    setIsWaitingForProjectLoad(false);
                } else {
                    // 等待数据加载
                    setIsWaitingForProjectLoad(true);
                    setShowAutoPopup(false);
                }
            }
        } else {
            // 没有项目名称，直接开始引导
            setShowTour(true);
            setShowAutoPopup(false);
            setIsWaitingForProjectLoad(false);
        }
    }, [navigate, location.pathname, checkProjectLoaded]);
    
    // 监听路由变化，等待页面加载完成
    useEffect(() => {
        if (isWaitingForProjectLoad && pendingProjectNameRef.current) {
            const expectedPath = `/projects/${pendingProjectNameRef.current}/runs`;
            const isOnTargetPath = location.pathname === expectedPath;
            
            
            if (isOnTargetPath) {
                // 开始检查数据是否加载完成
                clearCheckInterval();
                
                // 立即检查一次
                if (checkProjectLoaded()) {
                    setShowTour(true);
                    setIsWaitingForProjectLoad(false);
                    pendingProjectNameRef.current = null;
                    return;
                }
                
                // 轮询检查，最多等待10秒
                let checkCount = 0;
                const maxChecks = 50; // 50 * 200ms = 10秒
                
                checkIntervalRef.current = window.setInterval(() => {
                    checkCount++;
                    
                    if (checkProjectLoaded()) {
                        clearCheckInterval();
                        setShowTour(true);
                        setIsWaitingForProjectLoad(false);
                        pendingProjectNameRef.current = null;
                    } else if (checkCount >= maxChecks) {
                        clearCheckInterval();
                        setShowTour(true);
                        setIsWaitingForProjectLoad(false);
                        pendingProjectNameRef.current = null;
                    }
                }, 200);
                
                return () => clearCheckInterval();
            }
        }
    }, [location.pathname, isWaitingForProjectLoad, checkProjectLoaded, clearCheckInterval]);
    
    const closeAutoPopup = () => {
        setShowAutoPopup(false);
    };
    
    const completeTour = () => {
        setShowTour(false);
        setShowPostTourChoice(true);
        const state = {
            tourCompleted: true,
        };
        localStorage.setItem(GUIDE_STORAGE_KEY, JSON.stringify(state));
    };
    
    // 关闭引导但不标记为已完成，下次仍可进入
    const closeTourWithoutCompleting = () => {
        setShowTour(false);
        // 不修改localStorage，保持原有状态
    };

    const closePostTourChoice = () => {
        setShowPostTourChoice(false);
    };
    
    return (
        <FirstTimeGuideContext.Provider
            value={{
                showAutoPopup,
                showTour,
                showPostTourChoice,
                isWaitingForProjectLoad,
                startTour,
                closeAutoPopup,
                completeTour,
                closeTourWithoutCompleting,
                closePostTourChoice,
            }}
        >
            {children}
        </FirstTimeGuideContext.Provider>
    );
}

export const useFirstTimeGuide = () => {
    const context = useContext(FirstTimeGuideContext);
    if (!context) throw new Error('useFirstTimeGuide must be used within FirstTimeGuideProvider');
    return context;
};
