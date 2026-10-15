import type { Request, Response } from 'express';
import { TRPCError } from '@trpc/server';
import { experimentLogIdentitySchema } from '../../../shared/src/experiment';
import { AuthDao } from '../dao/Auth';
import { getRequestAuthToken } from '../trpc/router';
import { ExperimentService } from './experimentService';

/** Cookie/bearer authentication and ownership match the tRPC read endpoints. */
export async function downloadExperimentLog(req: Request, res: Response) {
    try {
        const input = experimentLogIdentitySchema.parse(req.params);
        const user = await AuthDao.getUserByToken(getRequestAuthToken(req.headers));
        if (!user) { res.status(401).end(); return; }
        if (user.mustChangePassword) { res.status(403).end(); return; }
        const source = await ExperimentService.businessLogSource(user, input.experimentId, input.attemptId, input.sourceId);
        const snapshot = source.nextOffset;
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${input.attemptId}-${input.sourceId}.txt"`);
        res.setHeader('Cache-Control', 'private, no-store');
        res.setHeader('X-Content-Type-Options', 'nosniff');
        for await (const chunk of ExperimentService.archivedLogChunks(source.id, snapshot)) {
            if (res.destroyed) break;
            if (!res.write(chunk.content)) {
                if (res.destroyed || res.writableEnded) break;
                await new Promise<void>(resolve => {
                    const done = () => { res.off('drain', done); res.off('close', done); resolve(); };
                    res.once('drain', done); res.once('close', done);
                });
                if (res.destroyed) break;
            }
        }
        res.end();
    } catch (error) {
        if (res.headersSent) { res.destroy(); return; }
        const status = error instanceof TRPCError && error.code === 'FORBIDDEN' ? 403
            : error && typeof error === 'object' && 'issues' in error ? 400 : 500;
        res.status(status).json({ error: status === 500 ? 'Log download failed' : 'Log is unavailable' });
    }
}
