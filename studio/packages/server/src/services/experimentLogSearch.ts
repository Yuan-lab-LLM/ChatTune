// Positions refer to UTF-16 characters inside archived, already redacted blocks.
// Runtime byte offsets are identities, never indexes into redacted strings.
export type LogPosition = { startOffset: number; characterOffset: number };
export type ArchivedLogChunk = { startOffset: number; endOffset: number; content: string };
export type LogMatch = { position: LogPosition; length: number; before: string; text: string; after: string };
export const suspectedLogErrors = ['Traceback', 'ERROR', 'CUDA out of memory'];
const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Bounded streaming scan. Keep enough lookahead for cross-block matches and excerpts. */
export async function searchArchivedLogs(chunks: AsyncIterable<ArchivedLogChunk>, input: {
    keywords: string[]; caseSensitive: boolean; cursor?: LogPosition; limit?: number; maxChunks?: number;
}) {
    const expression = new RegExp(input.keywords.map(escapeRegex).join('|'), input.caseSensitive ? 'gu' : 'giu');
    const lookahead = Math.max(...input.keywords.map(value => value.length)) + 100;
    const matches: LogMatch[] = [];
    let buffer = '', scanFrom = 0, consumed = 0, advanced = 0;
    let blocks: Array<{ startOffset: number; start: number; length: number }> = [];
    let endPosition: LogPosition | null = null;
    let firstPosition: LogPosition | undefined;
    const positionAt = (index: number): LogPosition => {
        const block = blocks.find(item => index < item.start + item.length) || blocks[blocks.length - 1];
        return { startOffset: block.startOffset, characterOffset: Math.min(block.length, index - block.start) };
    };
    const scan = (eof: boolean) => {
        let safeEnd = eof ? buffer.length : Math.max(0, buffer.length - lookahead);
        if (safeEnd > 0 && /[\uDC00-\uDFFF]/.test(buffer[safeEnd] || '')) safeEnd--;
        expression.lastIndex = scanFrom;
        let found: RegExpExecArray | null;
        while ((found = expression.exec(buffer))) {
            if (found.index >= safeEnd) break;
            matches.push({ position: positionAt(found.index), length: found[0].length,
                before: buffer.slice(Math.max(0, found.index - 100), found.index), text: found[0],
                after: buffer.slice(found.index + found[0].length, found.index + found[0].length + 100) });
            scanFrom = expression.lastIndex;
            if (matches.length >= (input.limit ?? 50)) return positionAt(scanFrom);
        }
        scanFrom = Math.max(scanFrom, safeEnd);
        return null;
    };
    for await (const chunk of chunks) {
        firstPosition ??= input.cursor || { startOffset: chunk.startOffset, characterOffset: 0 };
        blocks.push({ startOffset: chunk.startOffset, start: buffer.length, length: chunk.content.length });
        buffer += chunk.content;
        endPosition = { startOffset: chunk.startOffset, characterOffset: chunk.content.length };
        if (consumed === 0 && input.cursor?.startOffset === chunk.startOffset) scanFrom = input.cursor.characterOffset;
        consumed++;
        const before = scanFrom;
        const next = scan(false);
        advanced += scanFrom - before;
        if (next) return { matches, nextCursor: next };
        if (consumed >= (input.maxChunks ?? 128) && (advanced >= 4096 || !buffer.length)) {
            // Resume before the unscanned tail, including a potential split keyword.
            const cursor = positionAt(scanFrom);
            if (cursor.startOffset !== firstPosition.startOffset || cursor.characterOffset !== firstPosition.characterOffset)
                return { matches, nextCursor: cursor };
        }
        const discard = Math.max(0, scanFrom - 100);
        if (discard) {
            buffer = buffer.slice(discard);
            blocks = blocks.filter(block => block.start + block.length > discard)
                .map(block => ({ ...block, start: block.start - discard }));
            scanFrom -= discard;
        }
    }
    const next = blocks.length ? scan(true) : null;
    return { matches, nextCursor: next && endPosition &&
        (next.startOffset !== endPosition.startOffset || next.characterOffset !== endPosition.characterOffset) ? next : null };
}
