import { memo, ReactNode, useId, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Image } from 'antd';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { SparklesIcon } from 'lucide-react';
import MarkdownRender from '@/components/chat/bubbles/MarkdownRender';

import {
    Base64Source,
    BlockType,
    ContentBlock,
    SourceType,
    ToolResultBlock,
    ToolUseBlock,
    URLSource,
} from '@shared/types';
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from '@/components/ui/accordion.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Label } from '@/components/ui/label';
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from '@/components/ui/tooltip.tsx';

/**
 * Props for the BubbleBlock component that renders different types of content blocks.
 */
interface Props {
    block: ContentBlock | string;
    markdown?: boolean;
}

type ParsedTextSegment =
    | {
          type: 'text';
          value: string;
      }
    | {
          type: 'thinking';
          value: string;
      };

const THINK_TAG_PATTERN = /<think>([\s\S]*?)<\/think>/g;

const parseThinkTags = (text: string): ParsedTextSegment[] => {
    const segments: ParsedTextSegment[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    THINK_TAG_PATTERN.lastIndex = 0;
    while ((match = THINK_TAG_PATTERN.exec(text)) !== null) {
        if (match.index > lastIndex) {
            segments.push({
                type: 'text',
                value: text.slice(lastIndex, match.index),
            });
        }

        segments.push({
            type: 'thinking',
            value: match[1],
        });
        lastIndex = THINK_TAG_PATTERN.lastIndex;
    }

    if (lastIndex === 0) {
        return [{ type: 'text', value: text }];
    }

    if (lastIndex < text.length) {
        segments.push({
            type: 'text',
            value: text.slice(lastIndex),
        });
    }

    return segments;
};

/**
 * Render text content with optional markdown support.
 * Falls back to plain text with proper word wrapping if markdown is disabled.
 */
const TextBlockDiv = ({
    text,
    markdown,
}: {
    text: string;
    markdown: boolean;
}) => {
    if (markdown) {
        return <MarkdownRender text={text} />;
    }
    return (
        <div className="flex w-full max-w-full break-all whitespace-pre-wrap m-0 text-[13px] leading-5">
            {text}
        </div>
    );
};

/**
 * Render thinking content with special styling and optional markdown support.
 * Displays with a left border and muted colors to distinguish from regular text.
 */
const ThinkingBlockDiv = ({ thinking }: { thinking: string }) => {
    const { t } = useTranslation();
    return (
        <CollapsibleBlockDiv
            title={t('common.thinking')}
            icon={<SparklesIcon size={13} stroke="var(--primary-500)" />}
            content={thinking}
            defaultOpen={false}
        />
    );
};

const TextWithThinkTagsDiv = ({
    text,
    markdown,
}: {
    text: string;
    markdown: boolean;
}) => {
    const segments = parseThinkTags(text);

    if (segments.length === 1 && segments[0].type === 'text') {
        return <TextBlockDiv text={text} markdown={markdown} />;
    }

    return (
        <div className="flex min-w-0 flex-col w-full max-w-full gap-y-1.5">
            {segments.map((segment, index) => {
                if (segment.type === 'thinking') {
                    return (
                        <ThinkingBlockDiv
                            key={`${segment.type}-${index}`}
                            thinking={segment.value}
                        />
                    );
                }

                if (!segment.value) {
                    return null;
                }

                return (
                    <TextBlockDiv
                        key={`${segment.type}-${index}`}
                        text={segment.value}
                        markdown={markdown}
                    />
                );
            })}
        </div>
    );
};

/**
 * Render image content from base64 or URL sources.
 * Supports both embedded base64 data and external URLs.
 */
const ImageBlockDiv = ({ source }: { source: Base64Source | URLSource }) => {
    let url: string;
    if (source.type === SourceType.BASE64) {
        url = `data:${source.media_type};base64,${source.data}`;
    } else if (source.type === SourceType.URL) {
        url = source.url;
    } else {
        return null;
    }
    return <Image width={150} key={url} src={url} alt={url} />;
};

/**
 * Render video content from base64 or URL sources.
 * Note: Currently renders as audio element - may need correction for actual video.
 */
const VideoBlockDiv = ({ source }: { source: Base64Source | URLSource }) => {
    let url: string;
    if (source.type === 'base64') {
        url = `data:${source.media_type};base64,${source.data}`;
    } else {
        url = source.url;
    }
    return <video key={url} controls src={url} />;
};

/**
 * Render audio content from base64 or URL sources.
 * Note: Currently renders as video element - may need correction for actual audio.
 */
const AudioBlockDiv = ({ source }: { source: Base64Source | URLSource }) => {
    let url: string;
    if (source.type === 'base64') {
        url = `data:${source.media_type};base64,${source.data}`;
    } else {
        url = source.url;
    }
    return <audio key={url} controls src={url} />;
};

/**
 * Render tool usage information in a collapsible panel.
 * Shows tool name and full JSON details with syntax highlighting.
 */
const ToolUseBlockDiv = ({ block }: { block: ToolUseBlock }) => {
    const { t } = useTranslation();
    return (
        <Accordion className="w-full" type="single" collapsible>
            <AccordionItem value="header">
                <AccordionTrigger className="flex flex-row text-[12px] px-3.5 py-2 w-full rounded-t-[14px] rounded-b-[0px] bg-[linear-gradient(180deg,#373748_0%,#2e2f3e_100%)] text-white [&>svg]:stroke-white hover:no-underline cursor-pointer data-[state=closed]:rounded-b-[14px] shadow-[0_14px_30px_-24px_rgba(15,23,42,0.65)]">
                    <span className="truncate">
                        {t('chat.title-using-tool')}
                        {block.name + ' ...'}
                    </span>
                </AccordionTrigger>
                <AccordionContent className="w-full">
                    <SyntaxHighlighter
                        language="js"
                        customStyle={{
                            cursor: 'default',
                            padding: '14px',
                            margin: 0,
                            background: 'rgba(248,250,252,0.96)',
                            borderRadius: '0 0 14px 14px',
                        }}
                    >
                        {JSON.stringify(block, null, 2)}
                    </SyntaxHighlighter>
                </AccordionContent>
            </AccordionItem>
        </Accordion>
    );
};

/**
 * Render tool execution results in a collapsible panel.
 * Supports switching between formatted content and raw JSON output.
 */
const ToolResultBlockDiv = ({ block }: { block: ToolResultBlock }) => {
    const { t } = useTranslation();
    const [displayRaw, setDisplayRaw] = useState<boolean>(false);
    const displayModeId = useId();

    return (
        <Accordion className="w-full max-w-full" type="single" collapsible>
            <AccordionItem value="header">
                <div className="flex flex-row items-center rounded-t-[14px] rounded-b-[0px] bg-[linear-gradient(180deg,#373748_0%,#2e2f3e_100%)] text-white shadow-[0_14px_30px_-24px_rgba(15,23,42,0.65)]">
                    <AccordionTrigger className="min-w-0 flex-1 flex-row text-[12px] px-3.5 py-2 text-white [&>svg]:stroke-white hover:no-underline cursor-pointer data-[state=closed]:rounded-b-[14px]">
                        <div className="min-w-0 truncate">
                            {t('chat.title-tool-result')}&nbsp;
                            {block.name}
                        </div>
                    </AccordionTrigger>
                    <div className="flex shrink-0 items-center gap-2 px-3.5 py-2">
                        <Label
                            className="truncate text-[11px] text-white/90"
                            htmlFor={displayModeId}
                        >
                            Display Raw
                        </Label>
                        <Switch
                            id={displayModeId}
                            checked={displayRaw}
                            onCheckedChange={(checked) => {
                                setDisplayRaw(checked);
                            }}
                        />
                    </div>
                </div>
                <AccordionContent className="w-full">
                    {displayRaw ? (
                        <SyntaxHighlighter
                            language="js"
                            customStyle={{
                                cursor: 'default',
                                padding: '14px',
                                margin: 0,
                                background: 'rgba(248,250,252,0.96)',
                                borderRadius: '0 0 14px 14px',
                            }}
                        >
                            {JSON.stringify(block, null, 2)}
                        </SyntaxHighlighter>
                    ) : (
                        <ToolResultRender output={block.output} />
                    )}
                </AccordionContent>
            </AccordionItem>
        </Accordion>
    );
};

/*
 * Render the output of a tool result block.
 *
 * @param output - The output content of the tool result block.
 *
 * @return JSX.Element representing the rendered output.
 */
const ToolResultRender = ({
    output,
}: {
    output: ToolResultBlock['output'];
}) => {
    if (typeof output === 'string') {
        return (
            <div className="w-full bg-[rgba(248,250,252,0.96)] p-3.5 rounded-b-[14px] border border-slate-200/60 border-t-0 [&>p]:mt-0!">
                <MarkdownRender text={'- ' + output} />
            </div>
        );
    }

    return (
        <div className="w-full bg-[rgba(248,250,252,0.96)] p-3.5 rounded-b-[14px] border border-slate-200/60 border-t-0 [&>p]:mt-0!">
            {output.map((block) => {
                switch (block.type) {
                    case BlockType.TEXT:
                        return <MarkdownRender text={'- ' + block.text} />;
                    case BlockType.IMAGE:
                        if (block.source.type === SourceType.BASE64) {
                            return (
                                <Image
                                    src={block.source.data}
                                    className="max-w-full max-h-[200px]"
                                />
                            );
                        } else if (block.source.type === SourceType.URL) {
                            return (
                                <Image
                                    src={block.source.url}
                                    className="max-w-full max-h-[200px]"
                                />
                            );
                        }
                        return null;
                    case BlockType.AUDIO:
                        if (block.source.type === SourceType.BASE64) {
                            return <audio src={block.source.data} controls />;
                        } else if (block.source.type === SourceType.URL) {
                            return <audio src={block.source.url} controls />;
                        }
                }
            })}
        </div>
    );
};

interface CollapsibleBlockDivProps {
    title: string;
    icon: ReactNode;
    tooltip?: string;
    content: ReactNode;
    defaultOpen?: boolean;
}

export const CollapsibleBlockDiv = ({
    title,
    content,
    icon,
    tooltip,
    defaultOpen = true,
}: CollapsibleBlockDivProps) => {
    return (
        <Accordion
            className="w-full"
            type="single"
            collapsible
            defaultValue={defaultOpen ? 'thinking' : undefined}
        >
            <AccordionItem value="thinking">
                <AccordionTrigger className="flex items-center [&>svg]:mb-1 border border-border h-7 max-w-fit px-2.5 py-1 text-muted-foreground text-[11px] hover:no-underline cursor-pointer">
                    {tooltip ? (
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div className="flex flex-row items-center gap-x-1 truncate text-[11px]">
                                    {icon}
                                    {title}
                                </div>
                            </TooltipTrigger>
                            <TooltipContent>{tooltip}</TooltipContent>
                        </Tooltip>
                    ) : (
                        <div className="flex h-full gap-x-1 items-center">
                            {icon}
                            {title}
                        </div>
                    )}
                </AccordionTrigger>
                <AccordionContent className="border-l border-border p-2.5 mt-1.5">
                    {content}
                </AccordionContent>
            </AccordionItem>
        </Accordion>
    );
};

/**
 * Main component that renders different types of content blocks in chat bubbles.
 * Supports text, thinking, media (image/video/audio), and tool-related content.
 */
const BubbleBlock = ({ block, markdown = true }: Props) => {
    if (typeof block === 'string') {
        return <TextWithThinkTagsDiv text={block} markdown={markdown} />;
    }

    switch (block.type) {
        case BlockType.TEXT:
            return <TextWithThinkTagsDiv text={block.text} markdown={markdown} />;
        case BlockType.THINKING:
            return <ThinkingBlockDiv thinking={block.thinking} />;
        case BlockType.IMAGE:
            return <ImageBlockDiv source={block.source} />;
        case BlockType.VIDEO:
            return <VideoBlockDiv source={block.source} />;
        case BlockType.AUDIO:
            return <AudioBlockDiv source={block.source} />;
        case BlockType.TOOL_USE:
            return <ToolUseBlockDiv block={block} />;
        case BlockType.TOOL_RESULT:
            return <ToolResultBlockDiv block={block} />;
    }
};

export default memo(BubbleBlock);
