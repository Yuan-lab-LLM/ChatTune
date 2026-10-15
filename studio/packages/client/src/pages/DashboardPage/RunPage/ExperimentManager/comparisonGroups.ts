/** Keep explicitly selected evaluations that are not already shown for a training run. */
export function standaloneComparisonGroups<T extends { result: { id: string } }>(
    groups: Record<string, T[]>, displayedResultIds: Iterable<string>,
): Record<string, T[]> {
    const displayed = new Set(displayedResultIds);
    return Object.fromEntries(Object.entries(groups)
        .map(([key, entries]) => [key, entries.filter(entry => !displayed.has(entry.result.id))] as const)
        .filter(([, entries]) => entries.length > 0));
}
