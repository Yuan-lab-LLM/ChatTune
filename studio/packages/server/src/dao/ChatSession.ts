import crypto from 'crypto';
import { ChatSessionTable } from '../models/ChatSession';

const createSessionId = () =>
    `s${Date.now().toString(36)}${crypto.randomBytes(4).toString('hex')}`;

export class ChatSessionDao {
    static async getOrCreate(userId: string, runId: string) {
        const existing = await ChatSessionTable.findOne({
            where: { userId, runId },
        });
        if (existing) {
            return existing;
        }

        await ChatSessionTable.createQueryBuilder()
            .insert()
            .into(ChatSessionTable)
            .values({
                id: crypto.randomUUID(),
                userId,
                runId,
                sessionId: createSessionId(),
                clearedAt: null,
            })
            .orIgnore()
            .execute();

        return ChatSessionTable.findOneByOrFail({ userId, runId });
    }

    static async reset(userId: string, runId: string) {
        const session = await this.getOrCreate(userId, runId);
        session.sessionId = createSessionId();
        session.clearedAt = new Date().toISOString();
        await session.save();
        return session;
    }
}
