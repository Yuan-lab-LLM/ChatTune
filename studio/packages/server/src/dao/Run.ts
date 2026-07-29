import { FindOptionsWhere, In, ObjectLiteral, SelectQueryBuilder } from 'typeorm';
import {
    InputRequestData,
    ProjectData,
    RunData,
    Status,
    TableData,
} from '../../../shared/src';
import { RunTable } from '../models/Run';
import { RunView } from '../models/RunView';
import { MessageTable } from '../models/Message';
import { checkProcessByPid } from '../utils';
import { SpanDao } from './Trace';
import { SafeAuthUser } from './Auth';
import { UserRole } from '../models/Auth';

const canSeeAllRuns = (user?: SafeAuthUser | null) =>
    user?.role === UserRole.ADMIN;

const applyRunVisibility = <T extends ObjectLiteral>(
    queryBuilder: SelectQueryBuilder<T>,
    user?: SafeAuthUser | null,
) => {
    if (canSeeAllRuns(user)) {
        return queryBuilder;
    }

    if (!user) {
        return queryBuilder.andWhere('1 = 0');
    }

    const scopedQuery = queryBuilder
        .andWhere('run.ownerUserId IS NULL')
        .andWhere('run.status IN (:...visibleStatuses)', {
            visibleStatuses: [Status.RUNNING, Status.PENDING],
        });
    return user.assignedNodeId
        ? scopedQuery.andWhere('run.nodeId = :assignedNodeId', {
              assignedNodeId: user.assignedNodeId,
          })
        : scopedQuery.andWhere('1 = 0');
};

const toRunData = (row: RunTable): RunData =>
    ({
        id: row.id,
        project: row.project,
        name: row.name,
        timestamp: row.timestamp,
        run_dir: row.run_dir,
        pid: row.pid,
        status: row.status,
        ownerUserId: row.ownerUserId ?? null,
        nodeId: row.nodeId ?? null,
    }) as RunData;

export class RunDao {
    static async getLatestRunSummary(user?: SafeAuthUser | null): Promise<{
        project: string;
        runId: string;
        runName: string;
        timestamp: string;
        status: Status;
    } | null> {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run');
            applyRunVisibility(queryBuilder, user);
            const latestRun = await queryBuilder
                    .orderBy('run.timestamp', 'DESC')
                    .limit(1)
                    .getOne();

            if (!latestRun) {
                return null;
            }

            return {
                project: latestRun.project,
                runId: latestRun.id,
                runName: latestRun.name,
                timestamp: latestRun.timestamp,
                status: latestRun.status,
            };
        } catch (error) {
            console.error('Error in getLatestRunSummary:', error);
            throw error;
        }
    }

    static async getLatestRunnableRunSummary(user?: SafeAuthUser | null): Promise<{
        project: string;
        runId: string;
        runName: string;
        timestamp: string;
        status: Status;
    } | null> {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run')
                .where('run.status IN (:...statuses)', {
                    statuses: [Status.RUNNING, Status.PENDING],
                });
            applyRunVisibility(queryBuilder, user);
            const latestRun = await queryBuilder
                .orderBy('run.timestamp', 'DESC')
                .limit(1)
                .getOne();

            if (!latestRun) {
                return null;
            }

            return {
                project: latestRun.project,
                runId: latestRun.id,
                runName: latestRun.name,
                timestamp: latestRun.timestamp,
                status: latestRun.status,
            };
        } catch (error) {
            console.error('Error in getLatestRunnableRunSummary:', error);
            throw error;
        }
    }

    static async getLatestSharedRunnableRunSummary(): Promise<{
        project: string;
        runId: string;
        runName: string;
        timestamp: string;
        status: Status;
    } | null> {
        const latestRun = await RunTable.createQueryBuilder('run')
            .where('run.ownerUserId IS NULL')
            .andWhere('run.status IN (:...statuses)', {
                statuses: [Status.RUNNING, Status.PENDING],
            })
            .orderBy('run.timestamp', 'DESC')
            .limit(1)
            .getOne();

        return latestRun
            ? {
                  project: latestRun.project,
                  runId: latestRun.id,
                  runName: latestRun.name,
                  timestamp: latestRun.timestamp,
                  status: latestRun.status,
              }
            : null;
    }

    static async getRunnableRunSummaryForNode(nodeId: string): Promise<{
        project: string;
        runId: string;
        runName: string;
        timestamp: string;
        status: Status;
    } | null> {
        const run = await RunTable.createQueryBuilder('run')
            .where('run.nodeId = :nodeId', { nodeId })
            .andWhere('run.status IN (:...statuses)', {
                statuses: [Status.RUNNING, Status.PENDING],
            })
            .orderBy('run.timestamp', 'DESC')
            .limit(1)
            .getOne();
        return run
            ? {
                  project: run.project,
                  runId: run.id,
                  runName: run.name,
                  timestamp: run.timestamp,
                  status: run.status,
              }
            : null;
    }

    static async isRunnableRunForNode(runId: string, nodeId: string): Promise<boolean> {
        const selected = await this.getRunnableRunSummaryForNode(nodeId);
        return selected?.runId === runId;
    }

    static async isSharedRunnableRun(runId: string): Promise<boolean> {
        const count = await RunTable.createQueryBuilder('run')
            .where('run.id = :runId', { runId })
            .andWhere('run.ownerUserId IS NULL')
            .andWhere('run.status IN (:...statuses)', {
                statuses: [Status.RUNNING, Status.PENDING],
            })
            .getCount();
        return count > 0;
    }

    static async doesProjectExist(project: string, user?: SafeAuthUser | null) {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run')
                .where('run.project = :project', { project });
            applyRunVisibility(queryBuilder, user);
            const run = await queryBuilder.getOne();
            return run !== null;
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async doesRunExist(runId: string, user?: SafeAuthUser | null): Promise<boolean> {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run')
                .where('run.id = :runId', { runId });
            applyRunVisibility(queryBuilder, user);
            const run = await queryBuilder.getOne();
            return run !== null;
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async addRun(runData: RunData) {
        try {
            const run = RunTable.create({
                ...runData,
                ownerUserId: runData.ownerUserId ?? undefined,
                nodeId: runData.nodeId ?? undefined,
            });
            await run.save();
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async keepOnlyRunnableRunForNode(nodeId: string, runId: string) {
        await RunTable.createQueryBuilder()
            .update()
            .set({ status: Status.DONE })
            .where('nodeId = :nodeId', { nodeId })
            .andWhere('id != :runId', { runId })
            .andWhere('status IN (:...statuses)', {
                statuses: [Status.RUNNING, Status.PENDING],
            })
            .execute();
    }

    static async getRunNodeId(runId: string): Promise<string | null> {
        const run = await RunTable.findOne({
            where: { id: runId },
            select: ['id', 'nodeId'],
        });
        return run?.nodeId ?? null;
    }

    static async getVisibleRunNodeId(runId: string, user?: SafeAuthUser | null): Promise<string | null> {
        const queryBuilder = RunTable.createQueryBuilder('run')
            .select(['run.id', 'run.nodeId'])
            .where('run.id = :runId', { runId });
        applyRunVisibility(queryBuilder, user);
        const run = await queryBuilder.getOne();
        return run?.nodeId ?? null;
    }

    static async getRunOwnerUserId(runId: string): Promise<string | null> {
        const run = await RunTable.findOne({
            where: { id: runId },
            select: ['id', 'ownerUserId'],
        });

        return run?.ownerUserId ?? null;
    }

    /**
     * Retrieve paginated projects with aggregated run statistics
     *
     * This method performs an optimized database query to fetch project data with:
     * - Count of runs by status (running, pending, finished)
     * - Total number of runs per project
     * - Project creation timestamp (earliest run timestamp)
     * - Support for pagination, sorting, and filtering
     *
     * @param pagination - Object containing page and pageSize
     * @param pagination.page - Current page number (1-based)
     * @param pagination.pageSize - Number of items per page
     *
     * @param sort - Optional sorting configuration
     * @param sort.field - Field to sort by (project, running, pending, finished, total, createdAt)
     * @param sort.order - Sort direction ('asc' or 'desc')
     *
     * @param filters - Optional filters for querying
     * @param filters.project - Project name filter (uses LIKE for partial matching)
     *
     * @returns Promise resolving to TableData structure containing:
     *   - list: Array of ProjectData objects
     *   - total: Total number of projects (before pagination)
     *   - page: Current page number
     *   - pageSize: Items per page
     *
     * @throws Error if database query fails
     *
     * @example
     * const result = await RunDao.getProjects(
     *   { page: 1, pageSize: 10 },
     *   { field: 'total', order: 'desc' },
     *   { project: 'agent' }
     * );
     * // Returns: { list: [...], total: 25, page: 1, pageSize: 10 }
     */
    static async getProjects(
        pagination: {
            page: number;
            pageSize: number;
        },
        sort?: {
            field: string;
            order: 'asc' | 'desc';
        },
        filters?: {
            [key: string]: unknown;
        },
        user?: SafeAuthUser | null,
    ): Promise<TableData<ProjectData>> {
        try {
            // Build base query with aggregations using parameterized queries
            let queryBuilder = RunTable.createQueryBuilder('run')
                .select('run.project', 'project')
                .addSelect(
                    'SUM(CASE WHEN run.status = :runningStatus THEN 1 ELSE 0 END)',
                    'running',
                )
                .addSelect(
                    'SUM(CASE WHEN run.status = :pendingStatus THEN 1 ELSE 0 END)',
                    'pending',
                )
                .addSelect(
                    'SUM(CASE WHEN run.status = :doneStatus THEN 1 ELSE 0 END)',
                    'finished',
                )
                .addSelect('COUNT(*)', 'total')
                .addSelect('MIN(run.timestamp)', 'createdAt')
                .groupBy('run.project')
                .setParameters({
                    runningStatus: Status.RUNNING,
                    pendingStatus: Status.PENDING,
                    doneStatus: Status.DONE,
                });
            applyRunVisibility(queryBuilder, user);

            // Apply filters using HAVING (since we're using GROUP BY)
            if (filters?.project) {
                queryBuilder = queryBuilder.andWhere(
                    'run.project LIKE :projectFilter',
                    {
                        projectFilter: `%${filters.project}%`,
                    },
                );
            }

            // Apply sorting
            const sortField = sort?.field || 'createdAt';
            const sortOrder = (sort?.order?.toUpperCase() || 'DESC') as
                | 'ASC'
                | 'DESC';

            switch (sortField) {
                case 'project':
                    queryBuilder.orderBy('run.project', sortOrder);
                    break;
                case 'running':
                case 'pending':
                case 'finished':
                case 'total':
                case 'createdAt':
                    queryBuilder.orderBy(sortField, sortOrder);
                    break;
                default:
                    queryBuilder.orderBy('createdAt', 'DESC');
            }

            // Clone query for count (before pagination)
            const countQuery = queryBuilder.clone();
            const totalResult = await countQuery.getRawMany();
            const total = totalResult.length;

            // Apply pagination
            const offset = (pagination.page - 1) * pagination.pageSize;
            queryBuilder.limit(pagination.pageSize).offset(offset);

            // Execute paginated query
            const result = await queryBuilder.getRawMany();

            // Map results to ProjectData type
            const list = result.map((row) => ({
                project: row.project,
                running: Number(row.running) || 0,
                pending: Number(row.pending) || 0,
                finished: Number(row.finished) || 0,
                total: Number(row.total) || 0,
                createdAt: row.createdAt,
            })) as ProjectData[];

            return {
                list,
                total,
                page: pagination.page,
                pageSize: pagination.pageSize,
            };
        } catch (error) {
            console.error('Error in getProjects:', error);
            throw error;
        }
    }

    static async getAllProjects(user?: SafeAuthUser | null): Promise<ProjectData[]> {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run')
                .select('run.project', 'project')
                .addSelect(
                    'SUM(CASE WHEN run.status = :running THEN 1 ELSE 0 END)',
                    'running',
                )
                .addSelect(
                    'SUM(CASE WHEN run.status = :pending THEN 1 ELSE 0 END)',
                    'pending',
                )
                .addSelect(
                    'SUM(CASE WHEN run.status = :finished THEN 1 ELSE 0 END)',
                    'finished',
                )
                .addSelect('MIN(run.timestamp)', 'createdAt')
                .addSelect('COUNT(*)', 'total')
                .groupBy('run.project')
                .setParameters({
                    running: Status.RUNNING,
                    pending: Status.PENDING,
                    finished: Status.DONE,
                });
            applyRunVisibility(queryBuilder, user);
            const result = await queryBuilder.getRawMany();

            return result.map(
                (row) =>
                    ({
                        project: row.project,
                        running: parseInt(row.running),
                        pending: parseInt(row.pending),
                        finished: parseInt(row.finished),
                        total: parseInt(row.total),
                        createdAt: row.createdAt,
                    }) as ProjectData,
            );
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    /*
     * Get all runs for a project
     */
    static async getAllProjectRuns(project: string, user?: SafeAuthUser | null) {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run')
                .where('run.project = :project', { project });
            applyRunVisibility(queryBuilder, user);
            const result = await queryBuilder
                .orderBy('run.timestamp', 'DESC')
                .getMany();

            return result.map(toRunData);
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async getRunData(runId: string, user?: SafeAuthUser | null) {
        try {
            const queryBuilder = RunTable.createQueryBuilder('run')
                .leftJoinAndSelect('run.replies', 'replies')
                .leftJoinAndSelect('replies.messages', 'messages')
                .leftJoinAndSelect('run.inputRequests', 'inputRequests')
                .where('run.id = :runId', { runId });
            applyRunVisibility(queryBuilder, user);
            const result = await queryBuilder.getOne();

            const spans = await SpanDao.getSpansByConversationId(runId);

            if (result) {
                return {
                    runData: toRunData(result),
                    inputRequests: result.inputRequests.map(
                        (row) =>
                            ({
                                requestId: row.requestId,
                                agentId: row.agentId,
                                agentName: row.agentName,
                                structuredInput: row.structuredInput,
                            }) as InputRequestData,
                    ),
                    replies: result.replies.map((row) => ({
                        replyId: row.replyId,
                        replyRole: row.replyRole,
                        replyName: row.replyName,
                        createdAt: row.createdAt,
                        finishedAt: row.finishedAt,
                        messages: row.messages.map((msg) => ({
                            id: msg.id,
                            name: msg.msg.name,
                            role: msg.msg.role,
                            content: msg.msg.content,
                            timestamp: msg.msg.timestamp,
                            metadata: msg.msg.metadata,
                        })),
                    })),
                    spans: spans,
                };
            } else {
                throw new Error(`Run with id ${runId} not found`);
            }
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async changeRunStatus(runId: string, newStatus: Status) {
        try {
            const run = await RunTable.findOne({ where: { id: runId } });

            if (run) {
                run.status = newStatus;
                await run.save();
            } else {
                throw new Error(`Run with id ${runId} not found`);
            }
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async updateRunStatusAtBeginning() {
        try {
            const runs = await RunTable.find({
                where: [{ status: Status.RUNNING }, { status: Status.PENDING }],
            });

            for (const run of runs) {
                const processExists = await checkProcessByPid(run.pid);
                if (!processExists) {
                    run.status = Status.DONE;
                    await run.save();
                }
            }
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async getRunViewData(user?: SafeAuthUser | null) {
        if (!canSeeAllRuns(user)) {
            const queryBuilder = RunTable.createQueryBuilder('run');
            applyRunVisibility(queryBuilder, user);
            const monthKeys = Array.from({ length: 12 }, (_, index) => {
                const date = new Date();
                date.setDate(1);
                date.setHours(0, 0, 0, 0);
                date.setMonth(date.getMonth() - index);
                return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            });
            const oldestMonth = monthKeys[monthKeys.length - 1];
            const monthlyRunsQueryBuilder = queryBuilder
                .clone()
                .select("strftime('%Y-%m', run.timestamp)", 'month')
                .addSelect('COUNT(*)', 'count')
                .andWhere('run.timestamp >= :oldestMonthStart', {
                    oldestMonthStart: `${oldestMonth}-01 00:00:00`,
                })
                .groupBy("strftime('%Y-%m', run.timestamp)");
            const [
                totalProjects,
                totalRuns,
                weekStats,
                monthStats,
                yearStats,
                recentProjects,
                monthlyRunRows,
            ] = await Promise.all([
                queryBuilder
                    .clone()
                    .select('COUNT(DISTINCT run.project)', 'count')
                    .getRawOne(),
                queryBuilder.clone().getCount(),
                queryBuilder
                    .clone()
                    .select('COUNT(DISTINCT run.project)', 'projects')
                    .addSelect('COUNT(*)', 'runs')
                    .andWhere(
                        "run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-7 days')",
                    )
                    .getRawOne(),
                queryBuilder
                    .clone()
                    .select('COUNT(DISTINCT run.project)', 'projects')
                    .addSelect('COUNT(*)', 'runs')
                    .andWhere(
                        "run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 month')",
                    )
                    .getRawOne(),
                queryBuilder
                    .clone()
                    .select('COUNT(DISTINCT run.project)', 'projects')
                    .addSelect('COUNT(*)', 'runs')
                    .andWhere(
                        "run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 year')",
                    )
                    .getRawOne(),
                queryBuilder
                    .clone()
                    .select('run.project', 'project')
                    .addSelect('MAX(run.timestamp)', 'lastUpdateTime')
                    .addSelect('COUNT(*)', 'runCount')
                    .groupBy('run.project')
                    .orderBy('lastUpdateTime', 'DESC')
                    .limit(4)
                    .getRawMany(),
                monthlyRunsQueryBuilder.getRawMany(),
            ]);
            const monthlyRunCountByMonth = new Map(
                monthlyRunRows.map((row) => [
                    row.month,
                    Number(row.count || 0),
                ]),
            );
            const monthlyRuns = monthKeys.map((month) => ({
                month,
                count: monthlyRunCountByMonth.get(month) ?? 0,
            }));

            return {
                totalProjects: Number(totalProjects?.count || 0),
                totalRuns: Number(totalRuns || 0),
                projectsWeekAgo: Number(weekStats?.projects || 0),
                runsWeekAgo: Number(weekStats?.runs || 0),
                projectsMonthAgo: Number(monthStats?.projects || 0),
                runsMonthAgo: Number(monthStats?.runs || 0),
                projectsYearAgo: Number(yearStats?.projects || 0),
                runsYearAgo: Number(yearStats?.runs || 0),
                monthlyRuns: JSON.stringify(monthlyRuns),
                recentProjects: recentProjects.map((project) => ({
                    name: project.project,
                    lastUpdateTime: project.lastUpdateTime,
                    runCount: parseInt(project.runCount),
                })),
            };
        }

        // Get run view data
        const runViewData = await RunView.find();
        // Search four projects that are updated most recently
        const recentProjectsQueryBuilder = RunTable.createQueryBuilder('run')
            .select('run.project', 'project')
            .addSelect('MAX(run.timestamp)', 'lastUpdateTime')
            .addSelect('COUNT(*)', 'runCount')
            // 按项目分组
            .groupBy('run.project');
        applyRunVisibility(recentProjectsQueryBuilder, user);
        const recentProjects = await recentProjectsQueryBuilder
            // 按最后更新时间降序排序
            .orderBy('lastUpdateTime', 'DESC')
            // 限制返回4个结果
            .limit(4)
            .getRawMany();

        return {
            ...runViewData[0],
            recentProjects: recentProjects.map((project) => ({
                name: project.project,
                lastUpdateTime: project.lastUpdateTime,
                runCount: parseInt(project.runCount),
            })),
        };
    }

    static async deleteRuns(runIds: string[], user?: SafeAuthUser | null) {
        try {
            const allowedRuns = canSeeAllRuns(user)
                ? runIds
                : (
                      await RunTable.find({
                          where: {
                              id: In(runIds),
                              ownerUserId: user?.id,
                          },
                          select: ['id'],
                      })
                  ).map((run) => run.id);

            if (allowedRuns.length > 0) {
                await SpanDao.deleteSpansByConversationIds(allowedRuns);
            }
            const conditions: FindOptionsWhere<RunTable> = {
                id: In(allowedRuns),
            };
            const result = await RunTable.delete(conditions);
            return result.affected;
        } catch (error) {
            console.error('Error deleting runs:', error);
            throw error;
        }
    }

    static async deleteProjects(projects: string[], user?: SafeAuthUser | null) {
        try {
            const runsToDelete = canSeeAllRuns(user)
                ? await RunTable.find({
                      where: { project: In(projects) },
                      select: ['id'],
                  })
                : await RunTable.find({
                      where: {
                          project: In(projects),
                          ownerUserId: user?.id,
                      },
                      select: ['id'],
                  });
            const runIds = runsToDelete.map((run) => run.id);

            if (runIds.length > 0) {
                await SpanDao.deleteSpansByConversationIds(runIds);
            }

            const conditions: FindOptionsWhere<RunTable> = {
                id: In(runIds),
            };
            const result = await RunTable.delete(conditions);
            return result.affected;
        } catch (error) {
            console.error('Error deleting projects:', error);
            throw error;
        }
    }

    /**
     * 将 Date 格式化为数据库存储格式：YYYY-MM-DD HH:mm:ss
     */
    private static formatDateForDB(date: Date): string {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        const seconds = String(date.getSeconds()).padStart(2, '0');
        return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
    }

    /**
     * 获取系统概览统计数据
     */
    static async getSystemOverviewStats(user?: SafeAuthUser | null) {
        try {
            // 获取今日、本周、本月消息数
            const now = new Date();
            const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const weekStart = new Date(now);
            weekStart.setDate(now.getDate() - now.getDay() + 1);
            weekStart.setHours(0, 0, 0, 0);
            const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);

            // 使用 QueryBuilder 查询 JSON 字段中的 timestamp
            // 使用与数据库相同的格式：YYYY-MM-DD HH:mm:ss
            const buildMessageStatsQuery = (timestamp: string) => {
                const queryBuilder = MessageTable.createQueryBuilder('message')
                    .innerJoin('message.runId', 'run')
                    .where("json_extract(message.msg, '$.timestamp') >= :timestamp", { 
                        timestamp,
                    });
                applyRunVisibility(queryBuilder, user);
                return queryBuilder.getCount();
            };

            const [today, thisWeek, thisMonth] = await Promise.all([
                buildMessageStatsQuery(this.formatDateForDB(todayStart)),
                buildMessageStatsQuery(this.formatDateForDB(weekStart)),
                buildMessageStatsQuery(this.formatDateForDB(monthStart)),
            ]);

            return {
                messageStats: {
                    today,
                    thisWeek,
                    thisMonth
                }
            };
        } catch (error) {
            console.error('Error getting system overview stats:', error);
            throw error;
        }
    }
}
