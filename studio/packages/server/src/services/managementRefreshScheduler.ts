import { ManagementBizType } from '../../../shared/src';
import { ConfigManager } from '../../../shared/src/config/server';
import { managementCacheService } from './managementCacheService';

const REFRESH_INTERVALS: Record<ManagementBizType, number> = {
    dataset: 10 * 60 * 1000,
    model: 15 * 60 * 1000,
    medicalTest: 10 * 60 * 1000,
    evaluationResult: 5 * 60 * 1000,
};

class ManagementRefreshScheduler {
    private readonly registeredContainers = new Map<
        ManagementBizType,
        Set<string>
    >();

    private readonly timers = new Map<ManagementBizType, NodeJS.Timeout>();

    start(): void {
        for (const bizType of Object.keys(
            REFRESH_INTERVALS,
        ) as ManagementBizType[]) {
            if (this.timers.has(bizType)) {
                continue;
            }

            const timer = setInterval(() => {
                void this.refreshAll(bizType);
            }, REFRESH_INTERVALS[bizType]);
            this.timers.set(bizType, timer);
        }
    }

    stop(): void {
        for (const timer of this.timers.values()) {
            clearInterval(timer);
        }
        this.timers.clear();
    }

    registerContainer(
        bizType: ManagementBizType,
        containerName: string,
        nodeId = 'local',
    ): void {
        const trimmedName =
            containerName.trim() ||
            ConfigManager.getInstance().getDefaultContainerName();
        const target = `${nodeId}\t${trimmedName}`;
        if (!this.registeredContainers.has(bizType)) {
            this.registeredContainers.set(bizType, new Set([target]));
            return;
        }

        this.registeredContainers.get(bizType)?.add(target);
    }

    private async refreshAll(bizType: ManagementBizType): Promise<void> {
        const containers = this.registeredContainers.get(bizType);
        if (!containers || containers.size === 0) {
            return;
        }

        const targetContainers = Array.from(
            this.registeredContainers.get(bizType) ?? [],
        );

        await Promise.all(
            targetContainers.map(async (target) => {
                const [nodeId, containerName] = target.split('\t', 2);
                try {
                    await managementCacheService.refreshCache(
                        bizType,
                        nodeId,
                        containerName,
                        'schedule',
                    );
                } catch (error) {
                    console.error(
                        `[ManagementRefreshScheduler] Failed to refresh ${bizType} for ${containerName}:`,
                        error instanceof Error ? error.message : error,
                    );
                }
            }),
        );
    }
}

export const managementRefreshScheduler = new ManagementRefreshScheduler();
