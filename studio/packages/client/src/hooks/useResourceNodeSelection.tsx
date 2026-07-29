import { Select } from 'antd';
import { ReactNode, useEffect, useState } from 'react';
import { trpc } from '@/api/trpc';
import { useTranslation } from 'react-i18next';
import { useAuth } from '@/context/AuthContext';

const STORAGE_KEY = 'medflow_resource_node_id';
const EVENT_NAME = 'medflow-resource-node-change';

export const useResourceNodeSelection = () => {
    const { isAdmin } = useAuth();
    const [nodeId, setNodeIdState] = useState(() => localStorage.getItem(STORAGE_KEY) || 'all');
    const nodesQuery = trpc.queryResourceNodes.useQuery(undefined, { refetchInterval: 30000 });
    const nodes = nodesQuery.data?.data || [];

    useEffect(() => {
        if (isAdmin || nodes.length === 0) return;
        if (nodeId === 'all' || !nodes.some((node) => node.id === nodeId)) {
            const value = nodes[0].id;
            localStorage.setItem(STORAGE_KEY, value);
            setNodeIdState(value);
            window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: value }));
        }
    }, [isAdmin, nodeId, nodes]);

    useEffect(() => {
        const listener = (event: Event) => setNodeIdState((event as CustomEvent<string>).detail);
        window.addEventListener(EVENT_NAME, listener);
        return () => window.removeEventListener(EVENT_NAME, listener);
    }, []);

    const setNodeId = (value: string) => {
        localStorage.setItem(STORAGE_KEY, value);
        setNodeIdState(value);
        window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: value }));
    };

    return { nodeId, setNodeId, nodes, isLoading: nodesQuery.isLoading, isAdmin };
};

export const ResourceNodeSelector = ({ extra }: { extra?: ReactNode }) => {
    const { t } = useTranslation();
    const { nodeId, setNodeId, nodes, isLoading, isAdmin } = useResourceNodeSelection();
    if (!isAdmin) return null;

    return (
        <div className="flex flex-nowrap items-center gap-1 border-b border-border/25 bg-muted/15 px-2 py-1.5">
            <span className="shrink-0 text-[10px] font-medium text-muted-foreground">
                Node
            </span>
            <Select
                size="small"
                value={nodeId}
                loading={isLoading}
                onChange={setNodeId}
                popupMatchSelectWidth={false}
                className="w-[86px] flex-none [&_.ant-select-selector]:!h-7 [&_.ant-select-selector]:!rounded-lg [&_.ant-select-selector]:!px-2 [&_.ant-select-selection-item]:!text-[11px] [&_.ant-select-selection-item]:!font-medium [&_.ant-select-selection-item]:!leading-[26px] [&_.ant-select-arrow]:!text-[10px]"
                options={[
                    ...(isAdmin ? [{ value: 'all', label: t('resourceNode.all') }] : []),
                    ...nodes.map((node) => ({
                        value: node.id,
                        label: node.status === 'offline'
                            ? t('resourceNode.offlineName', { name: node.name })
                            : node.name,
                        disabled: node.status === 'offline',
                    })),
                ]}
            />
            {extra}
        </div>
    );
};
