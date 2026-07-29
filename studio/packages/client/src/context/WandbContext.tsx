import { createContext, ReactNode, useContext, useState, useCallback } from 'react';

/**
 * Wandb link item interface
 * Represents a single wandb monitoring link with its process ID
 */
export interface WandbLinkItem {
    pid: string;
    url: string;
}

/**
 * Wandb info for a specific user
 */
interface UserWandbInfo {
    url: string | null;
    pending: boolean;
}

/**
 * Wandb context type definition
 */
interface WandbContextType {
    wandbLinks: WandbLinkItem[];
    addWandbLink: (pid: string, url: string) => void;
    addWandbLinks: (links: WandbLinkItem[]) => void;
    clearWandbLinks: () => void;
    hasWandbLinks: boolean;
    // User-specific wandb info
    getUserWandbInfo: (userId: string) => UserWandbInfo;
    setUserWandbUrlInfo: (userId: string, url: string | null, pending: boolean) => void;
    clearUserWandbUrlInfo: (userId: string) => void;
    hasUserWandbUrl: (userId: string) => boolean;
}

export const normalizeWandbUrl = (url: string): string =>
    url.trim().replace(/%60(?=($|[?#]))/gi, '').replace(/[`,).;，。；]+(?=($|[?#]))/g, '');

const WandbContext = createContext<WandbContextType | null>(null);

interface WandbProviderProps {
    children: ReactNode;
}

/**
 * WandbContext Provider
 * Manages wandb monitoring links extracted from chat messages
 * Links are stored in memory and cleared on page refresh
 * Now supports user-specific wandb info
 */
export function WandbProvider({ children }: WandbProviderProps) {
    const [wandbLinks, setWandbLinks] = useState<WandbLinkItem[]>([]);
    // Map to store wandb info per user
    const [userWandbMap, setUserWandbMap] = useState<Map<string, UserWandbInfo>>(new Map());

    /**
     * Add a single wandb link
     * If the same PID already exists, it will be updated with the new URL
     */
    const addWandbLink = useCallback((pid: string, url: string) => {
        setWandbLinks((prev) => {
            // Check if this PID already exists
            const existingIndex = prev.findIndex((link) => link.pid === pid);
            if (existingIndex >= 0) {
                // Update existing link
                const updated = [...prev];
                updated[existingIndex] = { pid, url: normalizeWandbUrl(url) };
                return updated;
            }
            // Add new link
            return [...prev, { pid, url: normalizeWandbUrl(url) }];
        });
    }, []);

    /**
     * Add multiple wandb links at once
     * Used when processing a message that contains multiple links
     */
    const addWandbLinks = useCallback((links: WandbLinkItem[]) => {
        if (links.length === 0) return;
        
        setWandbLinks((prev) => {
            const updated = [...prev];
            links.forEach((newLink) => {
                const existingIndex = updated.findIndex((link) => link.pid === newLink.pid);
                if (existingIndex >= 0) {
                    updated[existingIndex] = { ...newLink, url: normalizeWandbUrl(newLink.url) };
                } else {
                    updated.push({ ...newLink, url: normalizeWandbUrl(newLink.url) });
                }
            });
            return updated;
        });
    }, []);

    /**
     * Clear all wandb links
     * Called when switching conversations or explicitly clearing
     */
    const clearWandbLinks = useCallback(() => {
        setWandbLinks([]);
    }, []);

    /**
     * Get wandb info for a specific user
     */
    const getUserWandbInfo = useCallback((userId: string): UserWandbInfo => {
        return userWandbMap.get(userId) || { url: null, pending: false };
    }, [userWandbMap]);

    /**
     * Set wandb URL info for a specific user
     * @param userId - The user ID
     * @param url - The wandb URL or null
     * @param pending - Whether the URL is pending
     */
    const setUserWandbUrlInfo = useCallback((userId: string, url: string | null, pending: boolean) => {
        setUserWandbMap((prev) => {
            const newMap = new Map(prev);
            newMap.set(userId, { url: url ? normalizeWandbUrl(url) : null, pending });
            return newMap;
        });
    }, []);

    /**
     * Clear wandb URL info for a specific user
     */
    const clearUserWandbUrlInfo = useCallback((userId: string) => {
        setUserWandbMap((prev) => {
            const newMap = new Map(prev);
            newMap.delete(userId);
            return newMap;
        });
    }, []);

    /**
     * Check if a user has a wandb URL
     */
    const hasUserWandbUrl = useCallback((userId: string): boolean => {
        const info = userWandbMap.get(userId);
        return info !== undefined && info.url !== null && info.url !== '';
    }, [userWandbMap]);

    const value: WandbContextType = {
        wandbLinks,
        addWandbLink,
        addWandbLinks,
        clearWandbLinks,
        hasWandbLinks: wandbLinks.length > 0,
        // User-specific methods
        getUserWandbInfo,
        setUserWandbUrlInfo,
        clearUserWandbUrlInfo,
        hasUserWandbUrl,
    };

    return (
        <WandbContext.Provider value={value}>
            {children}
        </WandbContext.Provider>
    );
}

/**
 * Hook to use wandb context
 * Must be used within a WandbProvider
 */
export function useWandb() {
    const context = useContext(WandbContext);
    if (!context) {
        throw new Error('useWandb must be used within a WandbProvider');
    }
    return context;
}



