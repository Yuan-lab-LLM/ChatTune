import type { ExperimentListFilters } from '../../../shared/src/experiment';

type FilterQuery<T> = { andWhere(condition: string, parameters?: Record<string, unknown>): T };

/** Filter before counting/paging. EXISTS avoids duplicate experiments with several matching tasks. */
export function applyExperimentListFilters<T extends FilterQuery<T>>(query: T, ownerUserId: string, input: ExperimentListFilters): T {
    query.andWhere('experiment.ownerUserId = :experimentOwnerId', { experimentOwnerId: ownerUserId });
    const search = input.search?.trim();
    if (search) {
        // Treat wildcard characters as literal name/ID text; all user input remains bound parameters.
        const pattern = `%${search.toLowerCase().replace(/[!%_]/g, '!$&')}%`;
        query.andWhere("(LOWER(experiment.name) LIKE :experimentSearch ESCAPE '!' OR LOWER(experiment.id) LIKE :experimentSearch ESCAPE '!')",
            { experimentSearch: pattern });
    }
    if (input.status) query.andWhere('experiment.status = :experimentStatus', { experimentStatus: input.status });
    if (input.tag) query.andWhere(`EXISTS (SELECT 1 FROM json_each(experiment.tags) WHERE json_each.value = :experimentTag)`,
        { experimentTag: input.tag });
    if (input.project?.trim()) query.andWhere('experiment.project = :experimentProject', { experimentProject: input.project.trim() });
    if (input.conversationRunId) query.andWhere('experiment.conversationRunId = :experimentRun', { experimentRun: input.conversationRunId });
    if (input.taskKind) query.andWhere(`(EXISTS (
        SELECT 1 FROM "experiment_task" "filterTask"
        WHERE "filterTask"."experimentId" = "experiment"."id" AND "filterTask"."kind" = :experimentTaskKind
    ) OR (experiment.status = :experimentDraftStatus AND json_extract(experiment.spec, '$.kind') = :experimentTaskKind))`,
        { experimentTaskKind: input.taskKind, experimentDraftStatus: 'draft' });
    if (input.createdFrom) query.andWhere('experiment.createdAt >= :experimentCreatedFrom', { experimentCreatedFrom: new Date(input.createdFrom).toISOString() });
    if (input.createdBefore) query.andWhere('experiment.createdAt < :experimentCreatedBefore', { experimentCreatedBefore: new Date(input.createdBefore).toISOString() });
    return query;
}
