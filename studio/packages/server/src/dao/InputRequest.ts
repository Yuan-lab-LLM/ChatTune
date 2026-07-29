import { InputRequestTable } from '../models/InputRequest';
import { RunTable } from '../models/Run';
import { Status } from '../../../shared/src/types/messageForm';

export class InputRequestDao {
    static async getInputRequestByRequestId(requestId: string) {
        try {
            return await InputRequestTable.createQueryBuilder('input_request')
                .select('input_request.requestId', 'requestId')
                .addSelect('input_request.agentId', 'agentId')
                .addSelect('input_request.agentName', 'agentName')
                .addSelect('input_request.structuredInput', 'structuredInput')
                .addSelect('input_request.run_id', 'runId')
                .where('input_request.requestId = :requestId', { requestId })
                .getRawOne<{
                    requestId: string;
                    agentId: string;
                    agentName: string;
                    structuredInput: Record<string, unknown> | null;
                    runId: string;
                }>();
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async saveInputRequest(inputRequest: {
        requestId: string;
        runId: string;
        agentId: string;
        agentName: string;
        structuredInput: Record<string, unknown> | null;
    }) {
        try {
            const newInputRequest = InputRequestTable.create({
                ...inputRequest,
            });
            await newInputRequest.save();
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async deleteInputRequest(inputRequestId: string) {
        try {
            const inputRequest = await InputRequestTable.createQueryBuilder(
                'input_request',
            )
                .where('input_request.requestId = :inputRequestId', {
                    inputRequestId,
                })
                .getOne();

            if (inputRequest) {
                await inputRequest.remove();
            } else {
                throw new Error(
                    `InputRequest with id ${inputRequestId} not found`,
                );
            }
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async deleteInputRequestsByRunId(runId: string): Promise<number> {
        try {
            const inputRequests = await InputRequestTable.createQueryBuilder(
                'input_request',
            )
                .where('input_request.run_id = :runId', { runId })
                .getMany();

            if (inputRequests.length > 0) {
                await InputRequestTable.remove(inputRequests);
            }
            
            return inputRequests.length;
        } catch (error) {
            console.error(error);
            throw error;
        }
    }

    static async updateInputRequests() {
        // 遍历所有的input requests，当对应的run处于Finished状态时，删除input request
        try {
            const inputRequests =
                await InputRequestTable.createQueryBuilder(
                    'input_request',
                ).getMany();

            let deletedCount = 0;
            for (const inputRequest of inputRequests) {
                const runId = inputRequest.runId;
                const run = await RunTable.findOne({ where: { id: runId } });
                if (run && run.status === Status.DONE) {
                    await InputRequestDao.deleteInputRequest(
                        inputRequest.requestId,
                    );
                    deletedCount++;
                }
            }
            
            if (deletedCount > 0) {
                console.log(`[InputRequest] Cleaned up ${deletedCount} orphaned input requests for finished runs`);
            }
        } catch (error) {
            console.error(error);
            throw error;
        }
    }
}
