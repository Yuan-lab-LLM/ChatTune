import { Modal, Button } from 'antd';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router-dom';
import { useFirstTimeGuide } from '@/context/FirstTimeGuideContext';

export function AutoPopup() {
    const { t, i18n } = useTranslation();
    const location = useLocation();
    const { showAutoPopup, startTour, closeAutoPopup } = useFirstTimeGuide();
    
    const isZh = i18n.language === 'zh';
    const isProjectRoute = /^\/projects\/[^/]+(?:\/.*)?$/.test(location.pathname);
    
    // 从 URL 解析项目名
    const projectName = location.pathname.match(/\/projects\/([^/]+)/)?.[1];
    
    const content = {
        title: isZh ? '🎉 欢迎使用 MedFlow ChatTune！' : '🎉 Welcome to MedFlow ChatTune!',
        layoutTitle: isZh ? '📍 页面布局' : '📍 Page Layout',
        layoutDesc: isZh 
            ? '• 左侧区域：功能导航和资源管理\n• 右侧区域：AI 对话和交互'
            : '• Left area: Navigation and resource management\n• Right area: AI conversation and interaction',
        quickStartTitle: isZh ? '🚀 引导流程' : '🚀 Tour Guide ',
        step1: isZh ? 'Step 1: 左侧功能栏介绍 - 五大模块导航' : 'Step 1: Left Sidebar - Five modules navigation',
        step2: isZh ? 'Step 2: Runs Tab - 点击运行实例进入对话' : 'Step 2: Runs Tab - Select running instance',
        step3: isZh ? 'Step 3: 对话页面 - 训练指标/对话输入/模板库' : 'Step 3: Conversation Page - Metrics/Input/Templates',
        step4: isZh ? 'Step 4: Data Tab - 查询可用数据集' : 'Step 4: Data Tab - Query datasets',
        step5: isZh ? 'Step 5: Model Tab - 查询可用模型' : 'Step 5: Model Tab - Query models',
        step6: isZh ? 'Step 6: Evaluation Tab - 查询可用评测集' : 'Step 6: Evaluation Tab - Query evaluations',
        
        startButton: isZh ? '开始引导' : 'Start Tour',
        closeButton: isZh ? '关闭' : 'Close',
        hintText: isZh 
            ? '提示:点击"关闭"退出本次引导，下次进入继续；点击"开始引导"进入引导模式。'
            : 'Tip:Click "Close" to exit and continue next time; click "Start Tour" to begin.',
    };
    
    const handleStartTour = () => {
        startTour(projectName);
    };
    
    return (
        <Modal
            title={<span className="text-lg font-medium">{content.title}</span>}
            open={showAutoPopup && isProjectRoute}
            footer={null}
            width={600}
            closable={false}
            maskClosable={false}
            centered
        >
            <div className="space-y-6 py-2">
                <div>
                    <h4 className="font-semibold text-foreground mb-2 text-sm">{content.layoutTitle}</h4>
                    <p className="text-muted-foreground/80 text-sm whitespace-pre-line leading-relaxed">
                        {content.layoutDesc}
                    </p>
                </div>
                
                <div>
                    <h4 className="font-semibold text-foreground mb-3 text-sm">{content.quickStartTitle}</h4>
                    <div className="text-muted-foreground/80 text-sm space-y-1.5 pl-1">
                        <p>{content.step1}</p>
                        <p>{content.step2}</p>
                        <p>{content.step3}</p>
                        <p>{content.step4}</p>
                        <p>{content.step5}</p>
                        <p>{content.step6}</p>
                    </div>
                </div>
                
                <p className="text-xs text-muted-foreground/70 text-center">
                    {content.hintText}
                </p>
            </div>
            
            <div className="pt-4 border-t border-border">
                <div className="flex items-center justify-end gap-2">
                    <Button onClick={() => closeAutoPopup()}>
                        {content.closeButton}
                    </Button>
                    <Button type="primary" onClick={handleStartTour}>
                        {content.startButton}
                    </Button>
                </div>
            </div>
        </Modal>
    );
}
