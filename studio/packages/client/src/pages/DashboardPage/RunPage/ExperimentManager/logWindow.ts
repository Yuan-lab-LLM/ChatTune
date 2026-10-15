export type LogChunk = { startOffset: number; endOffset: number; content: string };
export const LOG_WINDOW_CHUNKS = 64;
export const LOG_WINDOW_CHARACTERS = 512 * 1024;

/** Evict whole chunks so archived byte offsets remain valid for paging in either direction. */
export function mergeLogWindow(previous: LogChunk[], incoming: LogChunk[], direction: 'older' | 'newer'): LogChunk[] {
    if (!incoming.length) return previous;
    const all = new Map(previous.map(chunk => [chunk.startOffset, chunk]));
    for (const chunk of incoming) all.set(chunk.startOffset, chunk);
    const ordered = [...all.values()].sort((a, b) => a.startOffset - b.startOffset);
    const candidates = direction === 'older' ? ordered : ordered.reverse();
    const retained: LogChunk[] = [];
    let characters = 0;
    for (const chunk of candidates) {
        if (retained.length >= LOG_WINDOW_CHUNKS || characters + chunk.content.length > LOG_WINDOW_CHARACTERS) break;
        retained.push(chunk);
        characters += chunk.content.length;
    }
    if (direction === 'newer') retained.reverse();
    return retained.length === previous.length && retained.every((chunk, index) => chunk === previous[index])
        ? previous : retained;
}
