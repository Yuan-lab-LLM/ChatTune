import { z } from 'zod';

export const experimentIdSchema = z.string().min(1).max(120).regex(/^[A-Za-z0-9_-]+$/);
const jsonObject = z.record(z.unknown());
export const taskKindSchema = z.enum(['train', 'evaluate', 'preprocess', 'merge', 'publish', 'deploy', 'benchmark']);
export const attemptStatusSchema = z.enum(['prepared', 'starting', 'running', 'unknown', 'succeeded', 'failed', 'cancelled']);
export const experimentStatusSchema = z.enum(['draft', ...attemptStatusSchema.options]);
export const listExperimentsSchema = z.object({
    offset: z.number().int().nonnegative().default(0),
    limit: z.number().int().min(1).max(100).default(30),
    search: z.string().trim().max(200).optional(),
    status: experimentStatusSchema.optional(),
    taskKind: taskKindSchema.optional(),
    project: z.string().trim().max(200).optional(),
    tag: z.string().trim().min(1).max(40).optional(),
    conversationRunId: z.string().max(200).optional(),
    createdFrom: z.string().datetime().optional(),
    createdBefore: z.string().datetime().optional(),
}).refine(input => !input.createdFrom || !input.createdBefore || Date.parse(input.createdFrom) < Date.parse(input.createdBefore), {
    message: 'Creation date range must end after it starts', path: ['createdBefore'],
});
export type ListExperimentsInput = z.infer<typeof listExperimentsSchema>;
export type ExperimentListFilters = Omit<ListExperimentsInput, 'offset' | 'limit'>;
export const experimentMetadataSchema = z.object({
    experimentId: experimentIdSchema,
    notes: z.string().max(10000),
    tags: z.array(z.string().trim().min(1).max(40)).max(20)
        .transform(tags => [...new Set(tags)]),
});
export const experimentLogIdentitySchema = z.object({
    experimentId: experimentIdSchema, attemptId: experimentIdSchema, sourceId: experimentIdSchema,
});
export const logPositionSchema = z.object({
    startOffset: z.number().int().nonnegative().safe(),
    // Redaction may expand a runtime block. The service checks the actual archived string length.
    characterOffset: z.number().int().nonnegative().safe(),
});
export const searchExperimentLogsSchema = experimentLogIdentitySchema.extend({
    keyword: z.string().min(1).max(200),
    caseSensitive: z.boolean().default(false),
    suspectedErrors: z.boolean().default(false),
    snapshotEndOffset: z.number().int().nonnegative().safe().optional(),
    cursor: logPositionSchema.optional(),
});
export const experimentLogContextSchema = experimentLogIdentitySchema.extend({
    position: logPositionSchema,
    matchLength: z.number().int().positive().max(200),
    snapshotEndOffset: z.number().int().nonnegative().safe(),
});
export const assetKindSchema = z.enum(['dataset', 'base_model', 'checkpoint', 'adapter', 'model', 'report', 'config']);
export const assetInputSchema = z.object({
    assetVersionId: experimentIdSchema,
    role: z.string().min(1).max(80),
});
export const prepareExperimentSchema = z.object({
    runtimeNodeId: experimentIdSchema,
    actorUserId: experimentIdSchema,
    requestKey: experimentIdSchema,
    experimentId: experimentIdSchema.optional(),
    taskId: experimentIdSchema.optional(),
    name: z.string().min(1).max(200),
    project: z.string().max(200).default('default'),
    conversationRunId: z.string().max(200).optional(),
    workflowId: experimentIdSchema.optional(),
    kind: taskKindSchema,
    spec: jsonObject,
    inputs: z.array(assetInputSchema).max(100).default([]),
    resumeFromVersionId: experimentIdSchema.optional(),
    resourceReservationId: experimentIdSchema.optional(),
    container: z.string().min(1).max(200),
});
export const assetRegistrationSchema = z.object({
    id: experimentIdSchema,
    kind: assetKindSchema,
    role: z.string().min(1).max(80),
    direction: z.enum(['input', 'output']),
    resourceId: z.string().max(200).optional(),
    logicalName: z.string().min(1).max(300),
    fingerprint: z.string().regex(/^[a-f0-9]{64}$/).optional(),
    integrity: z.enum(['verified', 'incomplete', 'missing']),
    location: z.object({ container: z.string().min(1).max(200), path: z.string().min(1).max(4096) }),
    manifest: jsonObject,
    metadata: jsonObject,
});
export const metricSchema = z.object({
    datasetVersionId: experimentIdSchema,
    subset: z.string().max(200).default('all'),
    name: z.string().min(1).max(200),
    value: z.number().finite(),
    sampleCount: z.number().int().nonnegative().nullable(),
    unit: z.string().max(50).default('ratio'),
    direction: z.enum(['higher', 'lower', 'neutral']),
});
export const evaluationSchema = z.object({
    id: experimentIdSchema,
    modelVersionId: experimentIdSchema,
    reportVersionId: experimentIdSchema.optional(),
    protocol: jsonObject,
    protocolComplete: z.boolean(),
    expectedCount: z.number().int().nonnegative().nullable(),
    succeededCount: z.number().int().nonnegative().nullable(),
    failedCount: z.number().int().nonnegative().nullable(),
    metrics: z.array(metricSchema).max(2000),
});
export const serviceOperationSchema = z.object({
    operationId: experimentIdSchema,
    operation: z.literal('stop_service'),
    serviceInstanceId: experimentIdSchema,
    serviceRunId: experimentIdSchema,
    status: z.enum(['requested', 'succeeded', 'failed']),
    actorUserId: experimentIdSchema.nullable(),
    trigger: z.enum(['manual', 'workflow', 'admin', 'system']),
    cause: z.enum(['stop', 'restart']).optional(),
    reservationId: experimentIdSchema.nullable(),
    releaseReady: z.boolean().nullable(),
    result: z.string().max(8000),
    returnCode: z.number().int().nullable(),
}).refine(data => data.status === 'requested' ? data.releaseReady === null
    : data.releaseReady === (data.status === 'succeeded'), { message: 'Stop outcome and process release confirmation must agree' });
export type ServiceOperation = z.infer<typeof serviceOperationSchema>;
export type ServiceOperationHistory = ServiceOperation & {
    id: string; attemptId: string; requestedAt: string; completedAt: string | null;
    reservationStatus: string | null; releasedAt: string | null;
};
export const experimentEventSchema = z.object({
    eventId: experimentIdSchema,
    attemptId: experimentIdSchema,
    sequence: z.number().int().positive(),
    occurredAt: z.string().datetime(),
    type: z.enum(['status', 'configuration', 'artifact', 'evaluation', 'service_operation', 'workflow_status']),
    payload: jsonObject,
});
export const appendExperimentEventsSchema = z.object({
    runtimeNodeId: experimentIdSchema,
    events: z.array(experimentEventSchema).min(1).max(100),
});
export const appendExperimentLogSchema = z.object({
    runtimeNodeId: experimentIdSchema,
    attemptId: experimentIdSchema,
    source: z.object({
        id: experimentIdSchema,
        label: z.string().min(1).max(200),
        kind: z.enum(['execution', 'operation', 'training', 'training_phase', 'data_processing',
            'evaluation_phase', 'evaluation_result', 'deployment', 'service']),
        status: z.enum(['streaming', 'complete', 'error']),
        error: z.string().max(2000).optional(),
    }),
    chunks: z.array(z.object({
        startOffset: z.number().int().nonnegative().safe(),
        endOffset: z.number().int().positive().safe(),
        content: z.string().max(32768),
    })).max(8),
});
export type PrepareExperimentInput = z.infer<typeof prepareExperimentSchema>;
export type ExperimentEventInput = z.infer<typeof experimentEventSchema>;
export type AppendExperimentLogInput = z.infer<typeof appendExperimentLogSchema>;
export type AssetRegistration = z.infer<typeof assetRegistrationSchema>;
export type EvaluationRegistration = z.infer<typeof evaluationSchema>;
