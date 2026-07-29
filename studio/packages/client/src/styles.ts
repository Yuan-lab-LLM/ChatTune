export const RemoveScrollBarStyle: Record<string, unknown> = {
    '&::WebkitScrollbar': {
        display: 'none',
    },
    // Firefox
    scrollbarWidth: 'none',
    // IE 和 Edge
    msOverflowStyle: 'none',
};

export const SingleLineEllipsisStyle: Record<string, unknown> = {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    wordBreak: 'break-all',
};
