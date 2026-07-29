import { Modal, Card, Space } from 'antd';
import { HelpCircle, ChevronRight } from 'lucide-react';

interface GuideModalProps {
    title: string;
    content: string;
    open: boolean;
    onClose: () => void;
}

interface GuideItem {
    text: string;
    level: number;
}

interface GuideSection {
    number: string;
    title: string;
    items: GuideItem[];
}

/**
 * 计算行的缩进级别（每2个空格为一级）
 */
function getIndentLevel(line: string): number {
    const match = line.match(/^(\s*)/);
    if (!match) return 0;
    const spaces = match[1].length;
    return Math.floor(spaces / 2);
}

/**
 * 解析指南内容为结构化数据，支持层级缩进
 */
function parseGuideContent(content: string): GuideSection[] {
    const lines = content.trim().split('\n');
    const sections: GuideSection[] = [];
    let currentSection: GuideSection | null = null;

    lines.forEach((line) => {
        const trimmedLine = line.trim();
        if (!trimmedLine) return;

        const indentLevel = getIndentLevel(line);

        // 检测主标题行（如 "1. 查询数据集"）
        const mainTitleMatch = trimmedLine.match(/^(\d+)\.\s*(.+)$/);
        if (mainTitleMatch && indentLevel === 0) {
            if (currentSection) {
                sections.push(currentSection);
            }
            currentSection = {
                number: mainTitleMatch[1],
                title: mainTitleMatch[2],
                items: [],
            };
        }
        // 检测子项（以 "-"、"•" 开头）
        else if (currentSection && (trimmedLine.startsWith('-') || trimmedLine.startsWith('•'))) {
            currentSection.items.push({
                text: trimmedLine.substring(1).trim(),
                level: indentLevel,
            });
        }
        // 其他内容（如 "文件格式要求"）
        else if (currentSection && trimmedLine) {
            currentSection.items.push({
                text: trimmedLine,
                level: indentLevel,
            });
        }
    });

    if (currentSection) {
        sections.push(currentSection);
    }

    return sections;
}

/**
 * 使用指南弹窗组件
 * 用于显示各管理功能的使用说明 - 苹果风格卡片式布局
 */
export function GuideModal({ title, content, open, onClose }: GuideModalProps) {
    const sections = parseGuideContent(content);

    return (
        <Modal
            title={
                <div className="flex items-center gap-1.5">
                    <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center">
                        <HelpCircle className="w-3.5 h-3.5 text-primary" />
                    </div>
                    <span className="font-semibold">{title}</span>
                </div>
            }
            open={open}
            onCancel={onClose}
            footer={null}
            width={640}
            className="guide-modal"
        >
            <Space direction="vertical" size="small" className="w-full pt-1">
                {sections.map((section, index) => (
                    <Card
                        key={index}
                        size="small"
                        className="guide-card border-border/50 hover:border-primary/30 hover:shadow-sm transition-all"
                        styles={{ body: { padding: 12 } }}
                    >
                        <div className="flex items-start gap-2.5">
                            {/* 步骤序号 */}
                            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center">
                                <span className="text-xs font-semibold text-primary">
                                    {section.number}
                                </span>
                            </div>
                            
                            {/* 内容区域 */}
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-2">
                                    <h4 className="font-medium text-sm text-foreground">
                                        {section.title}
                                    </h4>
                                </div>
                                
                                {section.items.length > 0 && (
                                    <ul className="space-y-1.5">
                                        {section.items.map((item, itemIndex) => (
                                            <li
                                                key={itemIndex}
                                                className="flex items-start gap-1.5 text-sm"
                                                style={{ 
                                                    paddingLeft: `${item.level * 12}px`,
                                                }}
                                            >
                                                {item.level === 0 ? (
                                                    <ChevronRight className="w-3.5 h-3.5 text-primary/60 flex-shrink-0 mt-0.5" />
                                                ) : (
                                                    <span className="w-1.5 h-1.5 rounded-full bg-primary/40 flex-shrink-0 mt-1.5" />
                                                )}
                                                <span className={`leading-relaxed ${item.level === 0 ? 'text-muted-foreground' : 'text-muted-foreground/80'}`}>
                                                    {item.text}
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        </div>
                    </Card>
                ))}
            </Space>
        </Modal>
    );
}

export default GuideModal;
