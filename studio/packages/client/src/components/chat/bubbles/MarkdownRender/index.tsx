import { memo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { CheckIcon, CopyIcon } from 'lucide-react';
import { copyToClipboard } from '@/utils/common.ts';

import remarkGfm from 'remark-gfm';
import { Button } from '@/components/ui/button.tsx';
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from '@/components/ui/tooltip.tsx';
import { useTranslation } from 'react-i18next';

interface Props {
    text: string;
}

/**
 * Render markdown content with custom styling and code block support.
 * - Links: emphasized with underline and bold style
 * - Inline code: keeps inline flow with wrapping
 * - Code blocks: syntax highlighted with copy-to-clipboard header
 */
const MarkdownRender = ({ text }: Props) => {
    return (
        <div className="w-full max-w-full min-w-0 break-all">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                p: ({ ...props }) => (
                    <p
                        className="max-w-full break-all text-[13px] leading-5 [&:not(:first-child)]:mt-3"
                        {...props}
                    />
                ),
                h1: ({ ...props }) => (
                    <h1
                        className="scroll-m-20 text-[16px] leading-6 font-extrabold tracking-tight text-balance my-2 first:mt-0"
                        {...props}
                    />
                ),
                h2: ({ ...props }) => (
                    <h2
                        className="scroll-m-20 text-sm leading-6 font-semibold tracking-tight my-1.5 first:mt-0"
                        {...props}
                    />
                ),
                h3: ({ ...props }) => (
                    <h3
                        className="scroll-m-20 text-[13px] leading-5 font-semibold tracking-tight my-1.5 first:mt-0"
                        {...props}
                    />
                ),
                h4: ({ ...props }) => (
                    <h4
                        className="scroll-m-20 text-[13px] leading-5 font-semibold tracking-tight my-1.5 first:mt-0"
                        {...props}
                    />
                ),
                h5: ({ ...props }) => (
                    <h5
                        className="scroll-m-20 text-[13px] leading-5 font-semibold tracking-tight my-1.5 first:mt-0"
                        {...props}
                    />
                ),
                h6: ({ ...props }) => (
                    <h6
                        className="scroll-m-20 text-[13px] leading-5 font-semibold tracking-tight my-1.5 first:mt-0"
                        {...props}
                    />
                ),
                ul: ({ className, ...props }) => {
                    const classes = `max-w-full break-all list-disc my-2 ml-5 text-[13px] leading-5 [&>li]:mt-1 ${className || ''}`;
                    return <ul className={classes} {...props} />;
                },
                ol: ({ className, ...props }) => {
                    const classes = `max-w-full break-all list-decimal my-2 ml-5 text-[13px] leading-5 [&>li]:mt-1 ${className || ''}`;
                    return <ol className={classes} {...props} />;
                },
                blockquote: ({ ...props }) => (
                    <blockquote
                        className="mt-3 border-l-2 pl-4 text-[13px] leading-5 italic"
                        {...props}
                    />
                ),
                table: ({ ...props }) => (
                    <table
                        className="table-auto w-full max-w-full my-2 border-collapse break-all"
                        {...props}
                    />
                ),
                thead: ({ ...props }) => (
                    <thead className="bg-gray-100" {...props} />
                ),
                tbody: ({ ...props }) => <tbody {...props} />,
                tr: ({ ...props }) => <tr {...props} />,
                th: ({ ...props }) => (
                    <th className="px-3 py-2 border break-all" {...props} />
                ),
                td: ({ ...props }) => (
                    <td className="px-3 py-2 border break-all" {...props} />
                ),
                img: ({ ...props }) => (
                    <img className="max-w-full rounded my-2" {...props} />
                ),
                hr: ({ ...props }) => (
                    <hr className="border-t my-4" {...props} />
                ),
                input: ({ ...props }) => (
                    // used by remark-gfm task lists; keep inputs disabled to avoid interaction in chat UI
                    <input
                        type="checkbox"
                        disabled
                        className="mr-2 align-middle"
                        {...props}
                    />
                ),
                a: function ({ ...props }) {
                    return (
                        <a
                            className="break-all"
                            style={{
                                textDecoration: 'underline',
                                fontWeight: 'bold',
                            }}
                            {...props}
                        />
                    );
                },
                code: function ({ className, children, node, ...props }) {
                    // First check if it's a code block with three backticks ```
                    if (
                        node &&
                        node.position &&
                        node.position.end.line > node.position.start.line
                    ) {
                        const match = /language-(\w+)/.exec(className || '');
                        const language = match ? match[1] : 'text';

                        return (
                            <div className="flex flex-col w-full max-w-full mt-1 mb-1">
                                <CodeHeader
                                    onCopyBtnClick={() =>
                                        copyToClipboard(String(children))
                                    }
                                    language={language}
                                />

                                <SyntaxHighlighter
                                    language={language}
                                    customStyle={{
                                        cursor: 'default',
                                        padding: '16px',
                                        margin: 0,
                                        maxWidth: '100%',
                                        overflowX: 'auto',
                                        background: 'var(--color-code-bg)',
                                        borderRadius: '0 0 8px 8px',
                                    }}
                                >
                                    {String(children)}
                                </SyntaxHighlighter>
                            </div>
                        );
                    } else {
                        return (
                            <code
                                className="inline whitespace-pre-wrap break-all font-semibold px-[0.3rem] py-[0.2rem] font-mono rounded"
                                {...props}
                            >
                                {children}
                            </code>
                        );
                    }
                },
                }}
            >
                {text}
            </ReactMarkdown>
        </div>
    );
};

export default memo(MarkdownRender);

interface CodeHeaderProps {
    language: string;
    onCopyBtnClick: () => Promise<boolean>;
}

/**
 * Header for code blocks that displays the language and a copy button.
 * Shows success feedback temporarily after copying.
 */
const CodeHeader = ({ language, onCopyBtnClick }: CodeHeaderProps) => {
    const [copyState, setCopyState] = useState<'wait' | 'success' | 'error'>(
        'wait',
    );
    const { t } = useTranslation();

    return (
        <div className="flex flex-row justify-between items-center pl-4 pr-2 bg-primary-700 rounded-t-lg h-8 text-white cursor-default">
            <span className="truncate text-sm">{language.toUpperCase()}</span>
            <Tooltip>
                <TooltipTrigger asChild>
                    <Button
                        className="text-white hover:bg-primary-700 h-7 w-7 cursor-pointer"
                        variant="ghost"
                        onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            if (copyState === 'wait') {
                                onCopyBtnClick()
                                    .then((success) => {
                                        if (success) {
                                            setCopyState('success');
                                            setTimeout(() => {
                                                setCopyState('wait');
                                            }, 2000);
                                        }
                                    })
                                    .catch(() => {
                                        setCopyState('error');
                                        setTimeout(() => {
                                            setCopyState('wait');
                                        }, 2000);
                                    });
                            }
                        }}
                    >
                        {copyState === 'success' ? (
                            <CheckIcon className="text-white" />
                        ) : copyState === 'wait' ? (
                            <CopyIcon className="text-white" />
                        ) : null}
                    </Button>
                </TooltipTrigger>
                <TooltipContent>
                    {t('tooltip.button.copy-to-clipboard')}
                </TooltipContent>
            </Tooltip>
        </div>
    );
};
