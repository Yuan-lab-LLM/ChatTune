import { ResourceNodeInfo } from '../../../shared/src';

export type ResourceNodeConfig = ResourceNodeInfo & {
    baseUrl: string;
    defaultContainer: string;
    resourceApiToken?: string;
};

const resolveDefaultContainer = () =>
    process.env.MEDFLOW_DEFAULT_CONTAINER?.trim() ||
    process.env.DEFAULT_DOCKER_CONTAINER?.trim() ||
    process.env.AGENT3_DEFAULT_DOCKER_CONTAINER?.trim() ||
    'agent3';

export class ResourceNodeRegistry {
    private readonly nodes: ResourceNodeConfig[];

    constructor(raw = process.env.MEDFLOW_RESOURCE_NODES) {
        if (!raw?.trim()) {
            this.nodes = [{
                id: 'local',
                name: 'Local Node',
                baseUrl: process.env.MEDFLOW_LOCAL_RESOURCE_BASE_URL?.trim() || 'http://127.0.0.1:8099',
                defaultContainer: resolveDefaultContainer(),
                status: 'unknown',
            }];
            return;
        }
        const defaultContainer = resolveDefaultContainer();
        const parsed = JSON.parse(raw) as Array<Partial<ResourceNodeConfig>>;
        this.nodes = parsed.map((node) => {
            if (!node.id || !node.name || !node.baseUrl) {
                throw new Error('Each MEDFLOW_RESOURCE_NODES entry requires id, name and baseUrl');
            }
            return {
                id: node.id,
                name: node.name,
                baseUrl: node.baseUrl.replace(/\/+$/, ''),
                defaultContainer: node.defaultContainer?.trim() || defaultContainer,
                resourceApiToken: node.resourceApiToken?.trim() || undefined,
                status: 'unknown',
            };
        });
        if (new Set(this.nodes.map((node) => node.id)).size !== this.nodes.length) {
            throw new Error('MEDFLOW_RESOURCE_NODES contains duplicate node ids');
        }
        const dedicatedTokens = this.nodes
            .map((node) => node.resourceApiToken)
            .filter((token): token is string => !!token);
        if (new Set(dedicatedTokens).size !== dedicatedTokens.length) {
            throw new Error('MEDFLOW_RESOURCE_NODES contains duplicate dedicated resource API tokens');
        }
    }

    list(): Array<Omit<ResourceNodeConfig, 'resourceApiToken'>> {
        return this.nodes.map(({ resourceApiToken: _resourceApiToken, ...node }) => ({ ...node }));
    }

    get(nodeId: string): ResourceNodeConfig {
        const node = this.nodes.find((item) => item.id === nodeId);
        if (!node) throw new Error(`Unknown resource node: ${nodeId}`);
        return node;
    }
}

export const resourceNodeRegistry = new ResourceNodeRegistry();

export class RemoteResourceClient {
    private readonly fallbackToken = process.env.MEDFLOW_RESOURCE_API_TOKEN?.trim() || '';
    private readonly timeoutMs = Number(process.env.MEDFLOW_RESOURCE_TIMEOUT_MS || 60000);

    async request<T>(
        nodeId: string,
        path: string,
        init?: RequestInit,
        query?: Record<string, string | undefined>,
        timeoutMs = this.timeoutMs,
    ): Promise<T> {
        const node = resourceNodeRegistry.get(nodeId);
        const token = node.resourceApiToken || this.fallbackToken;
        if (!token) {
            throw new Error(`Resource API token is not configured for node ${nodeId}`);
        }
        const url = new URL(`${node.baseUrl}/internal/resources/${path}`);
        for (const [key, value] of Object.entries(query || {})) {
            if (value) url.searchParams.set(key, value);
        }
        const controller = new AbortController();
        let timedOut = false;
        const timer = setTimeout(() => {
            timedOut = true;
            controller.abort();
        }, timeoutMs);
        try {
            const response = await fetch(url, {
                ...init,
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'application/json',
                    ...(init?.headers || {}),
                },
                signal: controller.signal,
            });
            if (!response.ok) {
                throw new Error(`${node.name}: HTTP ${response.status} ${await response.text()}`);
            }
            return await response.json() as T;
        } catch (error) {
            if (timedOut) {
                throw new Error(
                    `资源节点 ${node.name} 请求超时：${init?.method || 'GET'} ${url.pathname}，超过 ${timeoutMs}ms`,
                );
            }
            throw error;
        } finally {
            clearTimeout(timer);
        }
    }
}

export const remoteResourceClient = new RemoteResourceClient();
