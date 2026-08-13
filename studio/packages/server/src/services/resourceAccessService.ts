import crypto from 'crypto';
import { EntityManager, In, MoreThan } from 'typeorm';
import { GPUInfo, ManagementBizType } from '../../../shared/src';
import { AuthUserTable, UserRole } from '../models/Auth';
import {
    GroupNodeAssignmentTable,
    PublicationStatus,
    ResourceAuditEventTable,
    ResourceCatalogTable,
    ResourcePublicationRequestTable,
    ResourceShareScopeTable,
    ResourceVisibility,
    UserGroupMemberTable,
    UserGroupTable,
} from '../models/ResourceAccess';
import type { SafeAuthUser } from '../dao/Auth';
import { remoteResourceClient, resourceNodeRegistry } from './resourceNodeService';
import {
    GroupResourceQuotaTable,
    TrainingReservationStatus,
    TrainingReservationTable,
} from '../models/TrainingResource';

const DEFAULT_GROUP_ID = 'default-users';
const SAFE_CONTAINER_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;

export class ResourceAccessService {
    static async ensureDefaultGroup() {
        let group = await UserGroupTable.findOne({ where: { id: DEFAULT_GROUP_ID } });
        if (!group) {
            group = UserGroupTable.create({
                id: DEFAULT_GROUP_ID,
                name: '默认用户组',
                description: '迁移和新建普通用户的默认资源组',
                defaultContainerName: process.env.AGENT3_DEFAULT_DOCKER_CONTAINER?.trim() || 'training_container',
                defaultEvaluateContainerName:
                    process.env.AGENT3_DEFAULT_EVALUATE_DOCKER_CONTAINER?.trim() || 'evaluation_container',
                defaultGrpoContainerName:
                    process.env.AGENT3_DEFAULT_GRPO_DOCKER_CONTAINER?.trim() || 'grpo_container',
            });
            await group.save();
        }
        const users = await AuthUserTable.find({ where: { role: UserRole.USER } });
        const memberships = await UserGroupMemberTable.find({
            where: { userId: In(users.map((user) => user.id)) },
        });
        const assigned = new Set(memberships.map((item) => item.userId));
        await UserGroupMemberTable.save(
            users.filter((user) => !assigned.has(user.id)).map((user) =>
                UserGroupMemberTable.create({
                    id: crypto.randomUUID(),
                    groupId: group!.id,
                    userId: user.id,
                }),
            ),
        );
    }

    static async getGroupForUser(userId: string) {
        const member = await UserGroupMemberTable.findOne({ where: { userId } });
        return member
            ? UserGroupTable.findOne({ where: { id: member.groupId } })
            : null;
    }

    static async getGroupById(groupId?: string | null) {
        const normalizedGroupId = groupId?.trim();
        return normalizedGroupId
            ? UserGroupTable.findOne({ where: { id: normalizedGroupId } })
            : null;
    }

    static async listGroups() {
        const [groups, members, users, nodeAssignments] = await Promise.all([
            UserGroupTable.find({ order: { name: 'ASC' } }),
            UserGroupMemberTable.find(),
            AuthUserTable.find({ order: { username: 'ASC' } }),
            GroupNodeAssignmentTable.find(),
        ]);
        const usernames = new Map(users.map((user) => [user.id, user.username]));
        return groups.map((group) => ({
            ...group,
            members: members
                .filter((member) => member.groupId === group.id)
                .map((member) => ({ userId: member.userId, username: usernames.get(member.userId) || member.userId })),
            nodeId: nodeAssignments.find((item) => item.groupId === group.id)?.nodeId || null,
            trainingContainerStatus: nodeAssignments.find((item) => item.groupId === group.id)?.trainingContainerStatus || null,
            trainingContainerError: nodeAssignments.find((item) => item.groupId === group.id)?.trainingContainerError || null,
            evaluationContainerStatus: nodeAssignments.find((item) => item.groupId === group.id)?.evaluationContainerStatus || null,
            evaluationContainerError: nodeAssignments.find((item) => item.groupId === group.id)?.evaluationContainerError || null,
            grpoContainerStatus: nodeAssignments.find((item) => item.groupId === group.id)?.grpoContainerStatus || null,
            grpoContainerError: nodeAssignments.find((item) => item.groupId === group.id)?.grpoContainerError || null,
        }));
    }

    static async createGroup(
        name: string,
        defaultContainerName: string,
        defaultEvaluateContainerName: string,
        defaultGrpoContainerName: string,
        description?: string,
    ) {
        const containerName = await this.validateAvailableContainerName(defaultContainerName);
        const evaluateContainerName = await this.validateAvailableContainerName(defaultEvaluateContainerName);
        const grpoContainerName = await this.validateAvailableContainerName(defaultGrpoContainerName);
        if (new Set([containerName, evaluateContainerName, grpoContainerName]).size !== 3) {
            throw new Error('auth.error.containerRolesMustDiffer');
        }
        const group = UserGroupTable.create({
            id: crypto.randomUUID(),
            name: name.trim(),
            description: description?.trim() || null,
            defaultContainerName: containerName,
            defaultEvaluateContainerName: evaluateContainerName,
            defaultGrpoContainerName: grpoContainerName,
        });
        return group.save();
    }

    static async setGroupContainer(groupId: string, defaultContainerName: string) {
        const group = await UserGroupTable.findOne({ where: { id: groupId } });
        if (!group) throw new Error('auth.error.groupNotFound');
        const assignment = await GroupNodeAssignmentTable.findOne({ where: { groupId } });
        const containerName = await this.validateAvailableContainerName(
            defaultContainerName,
            groupId,
            assignment?.nodeId,
        );
        if (containerName === group.defaultEvaluateContainerName || containerName === group.defaultGrpoContainerName) {
            throw new Error('auth.error.containerRolesMustDiffer');
        }
        if (assignment) await this.assertContainerReady(assignment.nodeId, containerName);
        group.defaultContainerName = containerName;
        await group.save();
        if (assignment) await this.validateGroupContainers(groupId, assignment);
        return group;
    }

    static async setGroupEvaluateContainer(groupId: string, defaultEvaluateContainerName: string) {
        const group = await UserGroupTable.findOne({ where: { id: groupId } });
        if (!group) throw new Error('auth.error.groupNotFound');
        const assignment = await GroupNodeAssignmentTable.findOne({ where: { groupId } });
        const containerName = await this.validateAvailableContainerName(
            defaultEvaluateContainerName,
            groupId,
            assignment?.nodeId,
        );
        if (containerName === group.defaultContainerName || containerName === group.defaultGrpoContainerName) {
            throw new Error('auth.error.containerRolesMustDiffer');
        }
        if (assignment) await this.assertContainerReady(assignment.nodeId, containerName);
        group.defaultEvaluateContainerName = containerName;
        await group.save();
        if (assignment) await this.validateGroupContainers(groupId, assignment);
        return group;
    }

    static async setGroupGrpoContainer(groupId: string, defaultGrpoContainerName: string) {
        const group = await UserGroupTable.findOne({ where: { id: groupId } });
        if (!group) throw new Error('auth.error.groupNotFound');
        const assignment = await GroupNodeAssignmentTable.findOne({ where: { groupId } });
        const containerName = await this.validateAvailableContainerName(
            defaultGrpoContainerName,
            groupId,
            assignment?.nodeId,
        );
        if (containerName === group.defaultContainerName || containerName === group.defaultEvaluateContainerName) {
            throw new Error('auth.error.containerRolesMustDiffer');
        }
        if (assignment) await this.assertContainerReady(assignment.nodeId, containerName);
        group.defaultGrpoContainerName = containerName;
        await group.save();
        if (assignment) await this.validateGroupContainers(groupId, assignment);
        return group;
    }

    private static validateContainerName(value: string) {
        const containerName = value.trim();
        if (!SAFE_CONTAINER_NAME.test(containerName)) {
            throw new Error('auth.error.containerNameInvalid');
        }
        return containerName;
    }

    private static async assertContainerReady(nodeId: string, container: string) {
        let result: { data: { exists: boolean; running: boolean } };
        try {
            result = await remoteResourceClient.request<{ data: { exists: boolean; running: boolean } }>(
                nodeId,
                'containers/status',
                { method: 'POST', body: JSON.stringify({ container }) },
            );
        } catch {
            throw new Error('auth.error.containerValidationFailed');
        }
        if (!result.data.exists) throw new Error('auth.error.containerNotFound');
        if (!result.data.running) throw new Error('auth.error.containerNotRunning');
    }

    private static async validateAvailableContainerName(
        defaultContainerName: string,
        groupId?: string,
        nodeId?: string,
    ) {
        const containerName = this.validateContainerName(defaultContainerName);
        if (!nodeId) return containerName;
        const assignments = await GroupNodeAssignmentTable.find({ where: { nodeId } });
        const otherGroupIds = assignments
            .map((assignment) => assignment.groupId)
            .filter((assignedGroupId) => assignedGroupId !== groupId);
        if (!otherGroupIds.length) return containerName;
        const existing = await UserGroupTable.findOne({
            where: [
                { id: In(otherGroupIds), defaultContainerName: containerName },
                { id: In(otherGroupIds), defaultEvaluateContainerName: containerName },
                { id: In(otherGroupIds), defaultGrpoContainerName: containerName },
            ],
        });
        if (existing) {
            throw new Error('auth.error.containerAlreadyBound');
        }
        return containerName;
    }

    static async getDefaultContainerForUser(user: SafeAuthUser) {
        if (user.role === UserRole.ADMIN) return null;
        return (await this.getGroupForUser(user.id))?.defaultContainerName || null;
    }

    static async getDefaultEvaluateContainerForUser(user: SafeAuthUser) {
        if (user.role === UserRole.ADMIN) return null;
        return (await this.getGroupForUser(user.id))?.defaultEvaluateContainerName || null;
    }

    static async getDefaultGrpoContainerForUser(user: SafeAuthUser) {
        if (user.role === UserRole.ADMIN) return null;
        return (await this.getGroupForUser(user.id))?.defaultGrpoContainerName || null;
    }

    static async resolveContainerForUser(user: SafeAuthUser, requestedContainer: string, adminDefault: string) {
        if (user.role === UserRole.ADMIN) return requestedContainer.trim() || adminDefault;
        const groupContainer = await this.getDefaultContainerForUser(user);
        if (!groupContainer) throw new Error('auth.error.groupNotFound');
        if (requestedContainer.trim() && requestedContainer.trim() !== groupContainer) {
            throw new Error('auth.error.resourceForbidden');
        }
        return groupContainer;
    }

    static async resolveEvaluateContainerForUser(user: SafeAuthUser, requestedContainer: string, adminDefault: string) {
        if (user.role === UserRole.ADMIN) return requestedContainer.trim() || adminDefault;
        const groupContainer = await this.getDefaultEvaluateContainerForUser(user);
        if (!groupContainer) throw new Error('auth.error.groupNotFound');
        if (requestedContainer.trim() && requestedContainer.trim() !== groupContainer) {
            throw new Error('auth.error.resourceForbidden');
        }
        return groupContainer;
    }

    static async resolveGrpoContainerForUser(user: SafeAuthUser, requestedContainer: string, adminDefault: string) {
        if (user.role === UserRole.ADMIN) return requestedContainer.trim() || adminDefault;
        const groupContainer = await this.getDefaultGrpoContainerForUser(user);
        if (!groupContainer) throw new Error('auth.error.groupNotFound');
        if (requestedContainer.trim() && requestedContainer.trim() !== groupContainer) {
            throw new Error('auth.error.resourceForbidden');
        }
        return groupContainer;
    }

    static async assertGroupExists(groupId: string) {
        if (!(await UserGroupTable.findOne({ where: { id: groupId } }))) {
            throw new Error('auth.error.groupNotFound');
        }
    }

    static async deleteGroup(groupId: string) {
        if (groupId === DEFAULT_GROUP_ID) throw new Error('auth.error.defaultGroupCannotDelete');
        if (await UserGroupMemberTable.count({ where: { groupId } })) {
            throw new Error('auth.error.groupNotEmpty');
        }
        if (await GroupNodeAssignmentTable.count({ where: { groupId } })) {
            throw new Error('auth.error.groupNodeAssigned');
        }
        if (await GroupResourceQuotaTable.count({ where: { groupId } })) {
            throw new Error('auth.error.groupQuotaAssigned');
        }
        if (await ResourceShareScopeTable.count({ where: { groupId } })) {
            throw new Error('auth.error.groupResourcesShared');
        }
        await UserGroupTable.delete({ id: groupId });
    }

    static async moveUser(userId: string, groupId: string) {
        const [user, group] = await Promise.all([
            AuthUserTable.findOne({ where: { id: userId } }),
            UserGroupTable.findOne({ where: { id: groupId } }),
        ]);
        if (!user || user.role !== UserRole.USER) throw new Error('auth.error.onlyUsersInGroups');
        if (!group) throw new Error('auth.error.groupNotFound');
        await UserGroupMemberTable.delete({ userId });
        return UserGroupMemberTable.create({ id: crypto.randomUUID(), userId, groupId }).save();
    }

    static async listNodeAssignments() {
        const assignments = await GroupNodeAssignmentTable.find();
        return resourceNodeRegistry.list().map((node) => ({
            id: node.id,
            name: node.name,
            groupIds: assignments
                .filter((item) => item.nodeId === node.id)
                .map((item) => item.groupId),
        }));
    }

    static async setGroupNode(groupId: string, nodeId: string | null) {
        const group = await UserGroupTable.findOne({ where: { id: groupId } });
        if (!group) {
            throw new Error('auth.error.groupNotFound');
        }
        const activeReservations = await TrainingReservationTable.count({
            where: {
                groupId,
                status: In([
                    TrainingReservationStatus.PREPARING,
                    TrainingReservationStatus.RESERVED,
                    TrainingReservationStatus.RUNNING,
                ]),
                expiresAt: MoreThan(new Date().toISOString()),
            },
        });
        if (activeReservations) throw new Error('auth.error.activeTrainingReservationExists');
        if (await GroupResourceQuotaTable.count({ where: { groupId } })) {
            throw new Error('auth.error.groupQuotaAssignedBeforeNodeChange');
        }
        if (nodeId) {
            resourceNodeRegistry.get(nodeId);
            await this.validateAvailableContainerName(group.defaultContainerName, groupId, nodeId);
            await this.validateAvailableContainerName(group.defaultEvaluateContainerName, groupId, nodeId);
            await this.validateAvailableContainerName(group.defaultGrpoContainerName, groupId, nodeId);
            await this.assertContainerReady(nodeId, group.defaultContainerName);
            await this.assertContainerReady(nodeId, group.defaultEvaluateContainerName);
            await this.assertContainerReady(nodeId, group.defaultGrpoContainerName);
        }
        await GroupNodeAssignmentTable.delete({ groupId });
        if (!nodeId) return null;
        const assignment = await GroupNodeAssignmentTable.create({
            id: crypto.randomUUID(),
            groupId,
            nodeId,
            trainingContainerStatus: 'pending',
            trainingContainerError: null,
            evaluationContainerStatus: 'pending',
            evaluationContainerError: null,
            grpoContainerStatus: 'pending',
            grpoContainerError: null,
        }).save();
        return this.validateGroupContainers(groupId, assignment);
    }

    static async validateGroupContainers(groupId: string, existingAssignment?: GroupNodeAssignmentTable) {
        const [group, assignment] = await Promise.all([
            UserGroupTable.findOne({ where: { id: groupId } }),
            existingAssignment
                ? Promise.resolve(existingAssignment)
                : GroupNodeAssignmentTable.findOne({ where: { groupId } }),
        ]);
        if (!group) throw new Error('auth.error.groupNotFound');
        if (!assignment) throw new Error('auth.error.groupNodeRequired');
        assignment.trainingContainerStatus = 'pending';
        assignment.trainingContainerError = null;
        assignment.evaluationContainerStatus = 'pending';
        assignment.evaluationContainerError = null;
        assignment.grpoContainerStatus = 'pending';
        assignment.grpoContainerError = null;
        await assignment.save();
        const validate = async (container: string) =>
            remoteResourceClient.request<{ data: { exists: boolean; running: boolean } }>(
                assignment.nodeId,
                'containers/status',
                { method: 'POST', body: JSON.stringify({ container }) },
            );
        const [training, evaluation, grpo] = await Promise.allSettled([
            validate(group.defaultContainerName),
            validate(group.defaultEvaluateContainerName),
            validate(group.defaultGrpoContainerName),
        ]);
        const apply = (
            result: PromiseSettledResult<{ data: { exists: boolean; running: boolean } }>,
            statusKey: 'trainingContainerStatus' | 'evaluationContainerStatus' | 'grpoContainerStatus',
            errorKey: 'trainingContainerError' | 'evaluationContainerError' | 'grpoContainerError',
        ) => {
            if (result.status === 'fulfilled' && result.value.data.exists && result.value.data.running) {
                assignment[statusKey] = 'ready';
                return;
            }
            assignment[statusKey] = 'failed';
            assignment[errorKey] = result.status === 'rejected'
                ? String(result.reason)
                : result.value.data.exists ? 'Docker 容器未运行' : 'Docker 容器不存在';
        };
        apply(training, 'trainingContainerStatus', 'trainingContainerError');
        apply(evaluation, 'evaluationContainerStatus', 'evaluationContainerError');
        apply(grpo, 'grpoContainerStatus', 'grpoContainerError');
        return assignment.save();
    }

    static async getAssignedNodeId(userId: string) {
        const member = await UserGroupMemberTable.findOne({ where: { userId } });
        if (!member) return null;
        const assignment = await GroupNodeAssignmentTable.findOne({ where: { groupId: member.groupId } });
        return assignment?.nodeId || null;
    }

    static async removeUserAccess(userId: string) {
        await UserGroupMemberTable.delete({ userId });
    }

    static async getVisibleNodeIds(user: SafeAuthUser) {
        if (user.role === UserRole.ADMIN) return null;
        const nodeId = user.assignedNodeId || await this.getAssignedNodeId(user.id);
        return new Set(nodeId ? [nodeId] : []);
    }

    static async assertNodeAccess(user: SafeAuthUser, nodeId: string) {
        if (user.role === UserRole.ADMIN) return;
        const visible = await this.getVisibleNodeIds(user);
        if (!visible?.has(nodeId)) throw new Error('auth.error.resourceForbidden');
    }

    static async filterGpus(user: SafeAuthUser, gpus: GPUInfo[]) {
        if (user.role === UserRole.ADMIN) return gpus;
        const assignedNodeId = user.assignedNodeId || await this.getAssignedNodeId(user.id);
        return gpus.filter((gpu) => assignedNodeId && gpu.nodeId === assignedNodeId);
    }

    static async filterResourceItems<T extends {
        nodeId?: string;
        type?: string;
        path?: string;
        name?: string;
        filename?: string;
        jobId?: string;
    }>(
        user: SafeAuthUser,
        bizType: ManagementBizType,
        containerName: string,
        items: T[],
    ) {
        if (user.role === UserRole.ADMIN) return items;
        const group = await this.getGroupForUser(user.id);
        const expectedContainer = bizType === 'medicalTest' || bizType === 'evaluationResult'
            ? group?.defaultEvaluateContainerName
            : group?.defaultContainerName;
        console.debug('[resource-debug] filterResourceItems start', {
            userId: user.id,
            username: user.username,
            bizType,
            requestedContainer: containerName,
            expectedContainer,
            groupId: group?.id || null,
            inputCount: items.length,
        });
        if (!group || containerName !== expectedContainer) {
            console.debug('[resource-debug] filterResourceItems blocked: container mismatch', {
                userId: user.id,
                bizType,
                requestedContainer: containerName,
                expectedContainer,
                hasGroup: Boolean(group),
            });
            throw new Error('auth.error.resourceForbidden');
        }
        const assignedNodeId = user.assignedNodeId || await this.getAssignedNodeId(user.id);
        const catalog = await ResourceCatalogTable.find({ where: { bizType, containerName } });
        const scopes = await this.getShareScopes(catalog.map((item) => item.id));
        const records = new Map(catalog.map((item) => [`${item.nodeId}:${item.itemKey}`, item]));
        const recordsByPath = new Map(
            catalog
                .filter((item) => item.sourcePath)
                .map((item) => [`${item.nodeId}:${item.sourcePath}`, item]),
        );
        const rejected: Array<Record<string, unknown>> = [];
        const filtered = items.reduce<T[]>((result, item) => {
            const key = item.name || item.filename || item.jobId || '';
            if (!assignedNodeId || item.nodeId !== assignedNodeId) {
                if (rejected.length < 5) {
                    rejected.push({
                        key,
                        itemNodeId: item.nodeId,
                        assignedNodeId,
                        reason: 'node mismatch',
                    });
                }
                return result;
            }
            const typedKey = item.type && key ? `${item.type}:${key}` : key;
            const sourcePath = item.path && key
                ? `${item.path.replace(/\/+$/, '')}/${key}`
                : '';
            const compatibleRecords = catalog.filter((entry) =>
                entry.nodeId === item.nodeId &&
                (entry.itemKey.endsWith(`:${key}`) || entry.sourcePath?.endsWith(`/${key}`)),
            );
            const record =
                records.get(`${item.nodeId}:${typedKey}`) ||
                records.get(`${item.nodeId}:${key}`) ||
                recordsByPath.get(`${item.nodeId}:${sourcePath}`);
            if (record) {
                const allowed = this.canReadResource(record, user.id, group.id, scopes.get(record.id));
                if (!allowed && rejected.length < 5) {
                    rejected.push({
                        key,
                        typedKey,
                        itemNodeId: item.nodeId,
                        reason: 'catalog record not readable',
                        visibility: record.visibility,
                        ownerUserId: record.ownerUserId,
                        groupId: record.groupId,
                    });
                }
                if (allowed) {
                    result.push({
                        ...item,
                        canDelete: record.ownerUserId === user.id,
                    } as T);
                }
                return result;
            }
            const compatibleRecord = compatibleRecords.length === 1
                ? compatibleRecords[0]
                : null;
            const allowedByCompatibleRecord = compatibleRecord
                ? this.canReadResource(
                    compatibleRecord,
                    user.id,
                    group.id,
                    scopes.get(compatibleRecord.id),
                )
                : false;
            if (!allowedByCompatibleRecord && rejected.length < 5) {
                rejected.push({
                    key,
                    typedKey,
                    itemNodeId: item.nodeId,
                    sourcePath,
                    compatibleRecordCount: compatibleRecords.length,
                    reason: 'no readable catalog match',
                });
            }
            if (allowedByCompatibleRecord && compatibleRecord) {
                result.push({
                    ...item,
                    canDelete: compatibleRecord.ownerUserId === user.id,
                } as T);
            }
            return result;
        }, []);
        console.debug('[resource-debug] filterResourceItems result', {
            userId: user.id,
            bizType,
            containerName,
            assignedNodeId,
            inputCount: items.length,
            outputCount: filtered.length,
            rejected,
        });
        return filtered;
    }

    static async assertResourceAccess(
        user: SafeAuthUser,
        bizType: ManagementBizType,
        containerName: string,
        itemKeyOrPath: string,
    ) {
        if (user.role === UserRole.ADMIN) return;
        const group = await this.getGroupForUser(user.id);
        const expectedContainer = bizType === 'medicalTest' || bizType === 'evaluationResult'
            ? group?.defaultEvaluateContainerName
            : group?.defaultContainerName;
        if (!group || containerName !== expectedContainer) {
            throw new Error('auth.error.resourceForbidden');
        }
        const assignedNodeId = user.assignedNodeId || await this.getAssignedNodeId(user.id);
        const resource = await ResourceCatalogTable.createQueryBuilder('resource')
            .where('resource.bizType = :bizType', { bizType })
            .andWhere('resource.containerName = :containerName', { containerName })
            .andWhere('resource.nodeId = :nodeId', { nodeId: assignedNodeId || '' })
            .andWhere(
                '(resource.itemKey = :value OR resource.itemKey LIKE :typedValue OR resource.sourcePath = :value OR resource.sourcePath LIKE :pathValue)',
                {
                    value: itemKeyOrPath,
                    typedValue: `%:${itemKeyOrPath}`,
                    pathValue: `%/${itemKeyOrPath}`,
                },
            )
            .getOne();
        if (!assignedNodeId || !resource || resource.nodeId !== assignedNodeId) {
            throw new Error('auth.error.resourceForbidden');
        }
        const scopes = await ResourceShareScopeTable.find({ where: { resourceId: resource.id } });
        if (!this.canReadResource(resource, user.id, group.id, scopes)) {
            throw new Error('auth.error.resourceForbidden');
        }
    }

    static async assertResourceWriteAccess(
        user: SafeAuthUser,
        bizType: ManagementBizType,
        containerName: string,
        itemKeyOrPath: string,
        allowMissing = false,
    ) {
        if (user.role === UserRole.ADMIN) return;
        const group = await this.getGroupForUser(user.id);
        const expectedContainer = bizType === 'medicalTest' || bizType === 'evaluationResult'
            ? group?.defaultEvaluateContainerName
            : group?.defaultContainerName;
        const assignedNodeId = user.assignedNodeId || await this.getAssignedNodeId(user.id);
        if (!group || !assignedNodeId || containerName !== expectedContainer) {
            throw new Error('auth.error.resourceForbidden');
        }
        const resource = await ResourceCatalogTable.createQueryBuilder('resource')
            .where('resource.bizType = :bizType', { bizType })
            .andWhere('resource.containerName = :containerName', { containerName })
            .andWhere('resource.nodeId = :nodeId', { nodeId: assignedNodeId })
            .andWhere(
                '(resource.itemKey = :value OR resource.itemKey LIKE :typedValue OR resource.sourcePath = :value OR resource.sourcePath LIKE :pathValue)',
                {
                    value: itemKeyOrPath,
                    typedValue: `%:${itemKeyOrPath}`,
                    pathValue: `%/${itemKeyOrPath}`,
                },
            )
            .getOne();
        if (!resource) {
            if (allowMissing) return;
            throw new Error('auth.error.resourceForbidden');
        }
        if (resource.ownerUserId !== user.id) {
            throw new Error('auth.error.resourceForbidden');
        }
    }

    static async recordResourceWrite(
        user: SafeAuthUser,
        bizType: ManagementBizType,
        nodeId: string,
        containerName: string,
        itemKey: string,
    ) {
        const existing = await ResourceCatalogTable.findOne({
            where: { bizType, nodeId, containerName, itemKey },
        });
        let resource: ResourceCatalogTable;
        if (user.role === UserRole.ADMIN) {
            resource = existing || await ResourceCatalogTable.create({
                    id: crypto.randomUUID(),
                    bizType,
                    nodeId,
                    containerName,
                    itemKey,
                    visibility: ResourceVisibility.PUBLIC,
                    ownerUserId: null,
                    groupId: null,
                }).save();
        } else {
            const group = await this.getGroupForUser(user.id);
            if (!group) throw new Error('auth.error.groupNotFound');
            if (existing?.ownerUserId && existing.ownerUserId !== user.id) {
                throw new Error('auth.error.resourceForbidden');
            }
            resource = await ResourceCatalogTable.create({
                ...existing,
                id: existing?.id || crypto.randomUUID(),
                bizType,
                nodeId,
                containerName,
                itemKey,
                visibility: ResourceVisibility.PRIVATE,
                ownerUserId: user.id,
                groupId: group.id,
            }).save();
        }
        await this.recordAuditEvent('resource_write', resource.id, user.id, {
            bizType, nodeId, containerName, itemKey, ownerUserId: resource.ownerUserId || null,
        });
        return resource;
    }

    static async removeResourceRecord(
        bizType: ManagementBizType,
        nodeId: string,
        containerName: string,
        itemKey: string,
        actorUserId?: string | null,
    ) {
        return ResourceCatalogTable.getRepository().manager.transaction(async (manager) => {
            const resource = await manager.findOne(ResourceCatalogTable, {
                where: { bizType, nodeId, containerName, itemKey },
            });
            if (!resource) return;
            const targets = resource.sourcePath
                ? await manager.find(ResourceCatalogTable, {
                    where: { bizType: resource.bizType, sourcePath: resource.sourcePath },
                })
                : [resource];
            const resourceIds = targets.map((target) => target.id);
            await this.recordAuditEvent('resource_delete', resource.id, actorUserId, {
                bizType,
                nodeId,
                containerName,
                itemKey,
                ownerUserId: resource.ownerUserId || null,
                resourceIds,
                sourcePath: resource.sourcePath || null,
            }, manager);
            await manager.delete(ResourceShareScopeTable, { resourceId: In(resourceIds) });
            await manager.delete(ResourcePublicationRequestTable, { resourceId: In(resourceIds) });
            await manager.delete(ResourceCatalogTable, { id: In(resourceIds) });
        });
    }

    static async listCatalog(user: SafeAuthUser) {
        const group = await this.getGroupForUser(user.id);
        const assignedNodeId = user.assignedNodeId || await this.getAssignedNodeId(user.id);
        const candidates = user.role === UserRole.ADMIN
            ? await ResourceCatalogTable.find({ order: { updatedAt: 'DESC' } })
            : await ResourceCatalogTable.createQueryBuilder('resource')
                .where('resource.nodeId = :nodeId', { nodeId: assignedNodeId || '' })
                .orderBy('resource.updatedAt', 'DESC')
                .getMany();
        const scopes = await this.getShareScopes(candidates.map((resource) => resource.id));
        const visible = user.role === UserRole.ADMIN
            ? candidates
            : candidates.filter((resource) =>
                !!group && this.canReadResource(resource, user.id, group.id, scopes.get(resource.id)));
        const resources = visible.map((resource) => ({
            ...resource,
            shareScopes: (scopes.get(resource.id) || []).map((scope) => ({
                groupId: scope.groupId || null,
                global: scope.scopeKey === '*',
            })),
        }));
        const requests = user.role === UserRole.ADMIN
            ? await ResourcePublicationRequestTable.find({ order: { updatedAt: 'DESC' } })
            : await ResourcePublicationRequestTable.find({ where: { requesterUserId: user.id }, order: { updatedAt: 'DESC' } });
        return { resources, requests };
    }

    private static async publishResourceWithManager(
        manager: EntityManager,
        resourceId: string,
        actorUserId?: string | null,
    ) {
        const resource = await manager.findOne(ResourceCatalogTable, { where: { id: resourceId } });
        if (!resource) throw new Error('auth.error.resourceNotFound');
        const targets = resource.sourcePath
            ? await manager.find(ResourceCatalogTable, { where: { bizType: resource.bizType, sourcePath: resource.sourcePath } })
            : [resource];
        await manager.delete(ResourceShareScopeTable, { resourceId: In(targets.map((target) => target.id)) });
        for (const target of targets) {
            target.visibility = ResourceVisibility.PUBLIC;
            target.groupId = null;
        }
        await manager.save(ResourceCatalogTable, targets);
        await this.recordAuditEvent('resource_published', resource.id, actorUserId, {
            resourceIds: targets.map((target) => target.id),
        }, manager);
        return { resource, resourceIds: targets.map((target) => target.id) };
    }

    static async publishResource(resourceId: string, adminId: string) {
        return ResourceCatalogTable.getRepository().manager.transaction((manager) =>
            this.publishResourceWithManager(manager, resourceId, adminId),
        );
    }

    static async requestPublication(resourceId: string, userId: string) {
        const resource = await ResourceCatalogTable.findOne({ where: { id: resourceId } });
        if (!resource || resource.ownerUserId !== userId || resource.visibility !== ResourceVisibility.PRIVATE) {
            throw new Error('auth.error.resourceForbidden');
        }
        const existing = await ResourcePublicationRequestTable.findOne({
            where: { resourceId, status: PublicationStatus.PENDING },
        });
        if (existing) return existing;
        const request = await ResourcePublicationRequestTable.create({
            id: crypto.randomUUID(), resourceId, requesterUserId: userId, status: PublicationStatus.PENDING,
        }).save();
        await this.recordAuditEvent('publication_requested', resourceId, userId, { requestId: request.id });
        return request;
    }

    static async reviewPublication(
        requestId: string,
        approved: boolean,
        adminId: string,
        note?: string,
    ) {
        return ResourceCatalogTable.getRepository().manager.transaction(async (manager) => {
            const request = await manager.findOne(ResourcePublicationRequestTable, { where: { id: requestId } });
            if (!request || request.status !== PublicationStatus.PENDING) throw new Error('auth.error.publicationRequestNotFound');
            request.status = approved ? PublicationStatus.APPROVED : PublicationStatus.REJECTED;
            request.reviewedBy = adminId;
            request.reviewNote = note?.trim() || null;
            if (approved) {
                await this.publishResourceWithManager(manager, request.resourceId, adminId);
            }
            await manager.save(ResourcePublicationRequestTable, request);
            await this.recordAuditEvent(
                approved ? 'publication_approved' : 'publication_rejected',
                request.resourceId,
                adminId,
                { requestId: request.id, note: request.reviewNote || null },
                manager,
            );
            return request;
        });
    }

    static async listAuditEvents() {
        return ResourceAuditEventTable.find({ order: { createdAt: 'DESC' }, take: 200 });
    }

    static async recordAuditEvent(
        eventType: string,
        resourceId?: string | null,
        actorUserId?: string | null,
        details?: Record<string, unknown> | null,
        manager?: EntityManager,
    ) {
        const event = (manager || ResourceAuditEventTable.getRepository().manager).create(ResourceAuditEventTable, {
            id: crypto.randomUUID(),
            eventType,
            resourceId: resourceId || null,
            actorUserId: actorUserId || null,
            details: details || null,
        });
        return (manager || ResourceAuditEventTable.getRepository().manager).save(ResourceAuditEventTable, event);
    }

    private static async getShareScopes(resourceIds: string[]) {
        if (!resourceIds.length) return new Map<string, ResourceShareScopeTable[]>();
        const scopes = await ResourceShareScopeTable.find({ where: { resourceId: In(resourceIds) } });
        const byResource = new Map<string, ResourceShareScopeTable[]>();
        for (const scope of scopes) {
            byResource.set(scope.resourceId, [...(byResource.get(scope.resourceId) || []), scope]);
        }
        return byResource;
    }

    private static canReadResource(
        resource: ResourceCatalogTable,
        userId: string,
        groupId: string,
        scopes?: ResourceShareScopeTable[],
    ) {
        if (resource.ownerUserId === userId) return true;
        if (resource.visibility !== ResourceVisibility.PUBLIC) return false;
        if (scopes?.length) {
            return scopes.some((scope) => scope.scopeKey === '*' || scope.groupId === groupId);
        }
        return !resource.groupId || resource.groupId === groupId;
    }
}
