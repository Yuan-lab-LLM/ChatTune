import { memo, ReactNode } from "react";
import { ContentType, Reply, TextBlock } from "@shared/types";
import BubbleBlock, {
  CollapsibleBlockDiv,
} from "@/components/chat/bubbles/BubbleBlock";
import { CircleAlertIcon, CopyIcon, CheckIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button.tsx";
import { useState } from "react";
import {
  AgentWaitingCard,
  cleanupAgentWaitingText,
  parseAgentWaitingPrompt,
} from "./AgentWaitingCard";
import { copyToClipboard } from "@/utils/common";
import type {
  AgentDatasetOption,
  AgentModelOption,
  AgentProtocol,
  AgentWaitingPrompt,
} from "./AgentWaitingCard";

interface Props {
  reply: Reply;
  avatar: ReactNode;
  markdown: boolean;
  onClick: (reply: Reply) => void;
  userAvatarRight: boolean;
  onAgentWaitingReply?: (text: string) => void;
  agentDatasets?: AgentDatasetOption[];
  agentModels?: AgentModelOption[];
  onRefreshAgentDatasets?: (
    containerName?: string,
  ) => Promise<AgentDatasetOption[]>;
  onRefreshAgentModels?: (
    containerName?: string,
  ) => Promise<AgentModelOption[]>;
  resourceGroupId?: string;
}

// 格式化时间为简短时间（用于显示）
const formatShortTime = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
};

// 提取消息文本内容
const extractMessageText = (content: ContentType): string => {
  if (typeof content === "string") {
    return content;
  }
  return content
    .map((block) => {
      if (block.type === "text") {
        return (block as TextBlock).text || "";
      }
      return "";
    })
    .join("\n");
};

const normalizeDisplayName = (name: string): string => {
  if (name.toLowerCase() === "system") {
    return "System";
  }

  const hasSessionSuffix = /#[a-z0-9]{6,}$/i.test(name);
  const withoutSession = name.replace(/#[a-z0-9]{6,}$/i, "");

  return withoutSession
    .replace(/_\[User-[A-Z0-9]+\]$/i, "")
    .replace(/\s*\[[^\]]+\]$/g, "") // 新增：去掉末尾的 [xxx] 格式
    .replace(hasSessionSuffix ? /_[^_]+$/ : /_admin$/i, "")
    .replace(/_admin$/i, "");
};

const extractMetadataProtocol = (
  metadata: object | null | undefined,
): AgentProtocol | null => {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  const protocol = (metadata as { protocol?: unknown }).protocol;
  if (!protocol || typeof protocol !== "object") {
    return null;
  }

  return protocol as AgentProtocol;
};

const hashAgentWaitingKeyPart = (value: string): string => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 0x01000193);
  }
  return (hash >>> 0).toString(36);
};

const buildAgentWaitingPersistenceKey = (
  reply: Reply,
  messageId: string | undefined,
  messageIndex: number,
  prompt: AgentWaitingPrompt,
): string => {
  const promptSignature = JSON.stringify({
    kind: prompt.kind,
    title: prompt.title,
    body: prompt.body,
    fields: prompt.fields,
    quickReplies: prompt.quickReplies,
    resourceContainer: prompt.resourceContainer,
    options: prompt.options,
  });
  const stableMessageId = messageId || `message-${messageIndex}`;
  return `agent-waiting-submitted:v1:${reply.replyId}:${stableMessageId}:${prompt.kind}:${hashAgentWaitingKeyPart(promptSignature)}`;
};

const INTERACTIVE_AGENT_NAMES = [
  "evaluator",
  "dataprocessor",
  "trainer",
  "inference",
  "monitor",
  "analysis",
];

const isInteractiveAgentName = (
  name: string | undefined | null,
  metadata?: object | null,
): boolean => {
  const normalizedName = (name || "").toLowerCase();
  if (
    INTERACTIVE_AGENT_NAMES.some((agentName) =>
      normalizedName.includes(agentName),
    )
  ) {
    return true;
  }

  // 如果 name 不匹配，检查 metadata.protocol.agent
  const protocol = extractMetadataProtocol(metadata);
  if (protocol?.agent) {
    const normalizedAgent = protocol.agent.toLowerCase();
    return INTERACTIVE_AGENT_NAMES.some((agentName) =>
      normalizedAgent.includes(agentName),
    );
  }

  return false;
};

const AsBubble = ({
  reply,
  avatar,
  markdown,
  onClick,
  userAvatarRight = false,
  onAgentWaitingReply,
  agentDatasets,
  agentModels,
  onRefreshAgentDatasets,
  onRefreshAgentModels,
  resourceGroupId,
}: Props) => {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const isUser = reply.replyRole.toLowerCase() === "user";
  const isAssistant = reply.replyRole.toLowerCase() === "assistant";
  const displayName = normalizeDisplayName(reply.replyName);
  const hasWaitingPrompt =
    isAssistant &&
    reply.messages.some(
      (msg) =>
        isInteractiveAgentName(msg.name || reply.replyName, msg.metadata) &&
        parseAgentWaitingPrompt(
          extractMessageText(msg.content),
          extractMetadataProtocol(msg.metadata),
        ),
    );

  // 根据角色确定布局方向
  const avatarRight = userAvatarRight && isUser;

  const renderBlock = (content: ContentType, markdown: boolean) => {
    if (typeof content === "string") {
      return (
        <BubbleBlock
          block={
            {
              type: "text",
              text: cleanupAgentWaitingText(content),
            } as TextBlock
          }
          markdown={markdown}
        />
      );
    }
    return content.map((block, index) => {
      if (block.type === "text") {
        return (
          <BubbleBlock
            key={`${block.type}-${index}`}
            block={{
              ...block,
              text: cleanupAgentWaitingText((block as TextBlock).text || ""),
            }}
            markdown={markdown}
          />
        );
      }
      return (
        <BubbleBlock
          key={`${block.type}-${index}`}
          block={block}
          markdown={markdown}
        />
      );
    });
  };

  // 复制所有消息内容（纯文本，不包含用户名）
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation(); // 防止触发 onClick

    // 合并所有 messages 的纯文本内容
    const content = reply.messages
      .map((msg) => extractMessageText(msg.content))
      .join("\n\n");

    try {
      const success = await copyToClipboard(content);
      if (!success) {
        throw new Error("Copy command was not accepted");
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // 获取最后一条消息的时间戳或 reply 的 createdAt
  const lastMessage = reply.messages[reply.messages.length - 1];
  const timestamp = lastMessage?.timestamp || reply.createdAt;

  // 气泡样式配置
  const bubbleStyles = isUser
    ? {
        // 用户消息样式 - 柔和蓝色高亮
        container: "flex-row-reverse",
        bubble:
          "bg-gradient-to-br from-[oklch(0.84_0.04_240)] via-[oklch(0.79_0.055_244)] to-[oklch(0.74_0.065_248)] text-white rounded-[22px] rounded-tr-md border border-sky-200/55 shadow-[0_16px_28px_-22px_rgba(59,130,246,0.35)] ml-auto",
        tail: "bubble-tail-right",
        nameColor: "text-primary",
        timeColor: "text-white/78",
        copyBtnColor: "text-white/75 hover:text-white hover:bg-white/18",
      }
    : {
        // AI 消息样式 - 浮层卡片感
        container: "flex-row",
        bubble:
          "bg-[linear-gradient(180deg,rgba(248,250,252,0.98)_0%,rgba(241,245,249,0.96)_100%)] text-foreground rounded-[22px] rounded-tl-md border border-slate-200/75 shadow-[0_18px_36px_-30px_rgba(15,23,42,0.22)]",
        tail: "bubble-tail-left",
        nameColor: "text-slate-600",
        timeColor: "text-slate-400",
        copyBtnColor:
          "text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10",
      };

  return (
    <div className="flex flex-col w-full max-w-full py-0.5">
      <div
        key={reply.replyId}
        className={`group flex ${bubbleStyles.container} gap-2.5 w-full max-w-full animate-fade-in-up`}
      >
        {/* 头像 */}
        <div className="flex-shrink-0">{avatar}</div>

        {/* 气泡内容区域 */}
        <div
          className={`flex min-w-0 flex-col flex-1 ${isUser ? "items-end" : "items-start"}`}
        >
          {/* 用户名 - 隐藏 cli_user */}
          {!isUser && (
            <div
              className={`mb-1 px-1 text-[10px] font-medium tracking-[0.01em] text-left ${bubbleStyles.nameColor}`}
            >
              {displayName}
            </div>
          )}

          {/* 气泡主体 */}
          <div
            className={`relative min-w-0 overflow-hidden break-all px-3.5 py-3 text-[13px] leading-[1.65] sm:px-4.5 sm:py-3.5 ${hasWaitingPrompt ? "max-w-[94%] sm:max-w-[90%]" : "max-w-[86%] sm:max-w-[76%]"} ${bubbleStyles.bubble} ${bubbleStyles.tail} cursor-pointer transition-all duration-200`}
            onClick={() => onClick(reply)}
          >
            {/* 消息内容 */}
            <div className="flex min-w-0 flex-col w-full max-w-full gap-y-1.5">
              {reply.messages.map((msg, index) => {
                const messageText = extractMessageText(msg.content);
                const waitingPrompt =
                  !isUser &&
                  isInteractiveAgentName(
                    msg.name || reply.replyName,
                    msg.metadata,
                  ) &&
                  parseAgentWaitingPrompt(
                    messageText,
                    extractMetadataProtocol(msg.metadata),
                  );
                if (msg.role.toLowerCase() === "user" && isAssistant) {
                  return (
                    <div key={index} className="animate-fade-in-up">
                      <CollapsibleBlockDiv
                        title={t("chat.title-hint-message")}
                        icon={<CircleAlertIcon size={12} />}
                        content={renderBlock(msg.content, markdown)}
                        tooltip={t("tooltip.header.hint-message")}
                      />
                    </div>
                  );
                }
                return (
                  <div key={index} className="animate-fade-in-up">
                    {waitingPrompt && (
                      <AgentWaitingCard
                        prompt={waitingPrompt}
                        onReply={onAgentWaitingReply}
                        persistenceKey={buildAgentWaitingPersistenceKey(
                          reply,
                          msg.id,
                          index,
                          waitingPrompt,
                        )}
                        datasets={agentDatasets}
                        models={agentModels}
                        onRefreshDatasets={onRefreshAgentDatasets}
                        onRefreshModels={onRefreshAgentModels}
                        resourceGroupId={resourceGroupId}
                      />
                    )}
                    {!waitingPrompt && renderBlock(msg.content, markdown)}
                  </div>
                );
              })}
            </div>

            {/* 底部工具栏：时间戳和复制按钮 */}
            <div
              className={`flex items-center gap-1.5 mt-2.5 ${isUser ? "justify-end" : "justify-start"}`}
            >
              <span className={`text-[10px] ${bubbleStyles.timeColor}`}>
                {formatShortTime(timestamp)}
              </span>

              <Button
                variant="ghost"
                size="icon"
                className={`h-4 w-4 opacity-0 group-hover:opacity-100 transition-all ${bubbleStyles.copyBtnColor}`}
                onClick={handleCopy}
                title={t("action.copy") || "Copy"}
              >
                {copied ? (
                  <CheckIcon className="h-2.5 w-2.5" />
                ) : (
                  <CopyIcon className="h-2.5 w-2.5" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(AsBubble);
