import { ExternalLinkIcon } from 'lucide-react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog.tsx';
import { Button } from '@/components/ui/button.tsx';
import { ScrollArea } from '@/components/ui/scroll-area.tsx';
import { WandbLinkItem } from '@/context/WandbContext.tsx';

interface WandbMonitorDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    wandbLinks: WandbLinkItem[];
}

/**
 * Wandb Monitor Dialog Component
 * Displays a list of wandb monitoring links with their process IDs
 * Users can click on a link to open it in a new tab
 */
export function WandbMonitorDialog({
    open,
    onOpenChange,
    wandbLinks,
}: WandbMonitorDialogProps) {
    const handleOpenLink = (url: string) => {
        window.open(url, '_blank', 'noopener,noreferrer');
    };

    // Truncate URL for display
    const truncateUrl = (url: string, maxLength: number = 50) => {
        if (url.length <= maxLength) return url;
        return url.substring(0, maxLength) + '...';
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[500px]">
                <DialogHeader>
                    <DialogTitle>Wandb 监控链接</DialogTitle>
                    <DialogDescription>
                        点击下方链接在新标签页中打开 wandb 监控页面
                    </DialogDescription>
                </DialogHeader>
                
                <div className="mt-4">
                    {wandbLinks.length === 0 ? (
                        <div className="text-center py-8 text-muted-foreground">
                            暂无 wandb 监控链接
                        </div>
                    ) : (
                        <ScrollArea className="h-[300px] pr-4">
                            <div className="space-y-3">
                                {wandbLinks.map((link, index) => (
                                    <div
                                        key={`${link.pid}-${index}`}
                                        className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                                    >
                                        <div className="flex-1 min-w-0 mr-4">
                                            <div className="font-medium text-sm">
                                                进程: {link.pid}
                                            </div>
                                            <div
                                                className="text-xs text-muted-foreground truncate"
                                                title={link.url}
                                            >
                                                {truncateUrl(link.url)}
                                            </div>
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => handleOpenLink(link.url)}
                                            className="shrink-0"
                                        >
                                            <ExternalLinkIcon className="h-4 w-4 mr-1" />
                                            打开
                                        </Button>
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>
                    )}
                </div>
                
                <div className="mt-4 flex justify-end">
                    <Button variant="outline" onClick={() => onOpenChange(false)}>
                        关闭
                    </Button>
                </div>
            </DialogContent>
        </Dialog>
    );
}
