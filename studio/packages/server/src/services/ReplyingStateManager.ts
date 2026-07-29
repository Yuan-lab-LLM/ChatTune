export class ReplyingStateManager {
    private static instance: ReplyingStateManager;
    private isReplying: boolean = false;

    private constructor() {}

    public static getInstance(): ReplyingStateManager {
        if (!ReplyingStateManager.instance) {
            ReplyingStateManager.instance = new ReplyingStateManager();
        }
        return ReplyingStateManager.instance;
    }

    public setReplyingState(state: boolean): void {
        this.isReplying = state;
    }

    public getReplyingState(): boolean {
        return this.isReplying;
    }
}
