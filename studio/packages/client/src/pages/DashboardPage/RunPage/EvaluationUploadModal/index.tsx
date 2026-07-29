import { memo, useState } from 'react';
import { Modal, Upload, Button, Form, message, Progress, Input, Alert } from 'antd';
import { useTranslation } from 'react-i18next';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload/interface';
import { useEnvironmentConfig } from '@/hooks/useEnvironmentConfig';

const { Dragger } = Upload;

interface Props {
    open: boolean;
    onCancel: () => void;
    onUpload: (params: {
        containerName: string;
        testType: string;
        filename: string;
        file: File;
    }) => Promise<void>;
    isUploading: boolean;
    uploadProgress: number;
}

const EvaluationUploadModal = ({
    open,
    onCancel,
    onUpload,
    isUploading,
    uploadProgress,
}: Props) => {
    const { t } = useTranslation();
    const { defaultEvaluateContainerName } = useEnvironmentConfig();
    const [form] = Form.useForm();
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [testType, setTestType] = useState('');
    const [fileError, setFileError] = useState<string>('');
    const [uploadError, setUploadError] = useState<string>('');

    const beforeUpload = (file: File): boolean => {
        setFileError('');
        
        // 验证文件类型
        const isJson = file.name.toLowerCase().endsWith('.json') || file.name.toLowerCase().endsWith('.jsonl');
        if (!isJson) {
            setFileError(t('evaluation.upload.file-type-error') || '只支持上传 .json 或 .jsonl 格式的文件');
            return false;
        }
        
        // 验证文件大小（20MB）
        const isLt20M = file.size / 1024 / 1024 < 20;
        if (!isLt20M) {
            setFileError(t('evaluation.upload.file-size-error') || '文件大小不能超过 20MB');
            return false;
        }
        
        return false; // 阻止自动上传
    };

    const handleUploadChange: UploadProps['onChange'] = ({ fileList: newFileList }) => {
        // 只保留最后一个文件
        if (newFileList.length > 1) {
            newFileList = [newFileList[newFileList.length - 1]];
        }
        setFileList(newFileList);
        
        // 清空错误信息当有新文件时
        if (newFileList.length > 0) {
            setFileError('');
        }
    };

    const handleUpload = async () => {
        // 验证文件列表
        if (fileList.length === 0) {
            setUploadError(t('evaluation.upload.error.no-file') || '请选择要上传的文件');
            return;
        }

        // 验证评测类型
        if (!testType.trim()) {
            setUploadError(t('evaluation.upload.error.test-type-required') || '请输入评测类型');
            return;
        }

        const file = fileList[0].originFileObj;
        if (!file) {
            setUploadError(t('evaluation.upload.error.invalid-file') || '文件对象无效');
            return;
        }

        try {
            setUploadError('');
            await onUpload({
                containerName: defaultEvaluateContainerName,
                testType: testType.trim(),
                filename: file.name,
                file,
            });
            // 上传成功后清空文件列表和评测类型
            setFileList([]);
            setTestType('');
            setUploadError('');
        } catch (error: any) {
            console.error('Upload failed:', error);
            const errorMsg = error?.message || t('evaluation.upload.error.default') || '上传失败，请重试';
            setUploadError(errorMsg);
        }
    };

    const handleCancel = () => {
        if (!isUploading) {
            setFileList([]);
            setTestType('');
            setFileError('');
            setUploadError('');
            onCancel();
        }
    };

    return (
        <Modal
            title={t('evaluation.upload.title') || '上传评测文件'}
            open={open}
            onCancel={handleCancel}
            footer={[
                <Button key="cancel" onClick={handleCancel} disabled={isUploading}>
                    {t('common.cancel') || '取消'}
                </Button>,
                <Button
                    key="upload"
                    type="primary"
                    onClick={handleUpload}
                    disabled={fileList.length === 0 || isUploading}
                    loading={isUploading}
                >
                    {isUploading ? t('common.uploading') || '上传中...' : t('common.upload') || '上传'}
                </Button>,
            ]}
            maskClosable={!isUploading}
            closable={!isUploading}
            className="upload-modal"
        >
            <Form form={form} layout="vertical" className="mt-4">
                <Form.Item
                    label={t('evaluation.upload.test-type') || '评测类型'}
                    required
                >
                    <Input
                        value={testType}
                        onChange={(e) => setTestType(e.target.value)}
                        placeholder={t('evaluation.upload.test-type-placeholder') || '请输入评测类型，如：中国执业医师资格考试'}
                        disabled={isUploading}
                    />
                </Form.Item>

                <Form.Item
                    label={t('evaluation.upload.file') || '评测文件'}
                    required
                >
                    <Dragger
                        name="file"
                        multiple={false}
                        fileList={fileList}
                        beforeUpload={beforeUpload}
                        onChange={handleUploadChange}
                        onRemove={() => {
                            setFileList([]);
                            setFileError('');
                        }}
                        disabled={isUploading}
                        showUploadList={{
                            showRemoveIcon: !isUploading,
                        }}
                    >
                        <p className="ant-upload-drag-icon">
                            <InboxOutlined />
                        </p>
                        <p className="ant-upload-text">
                            {t('evaluation.upload.drag') || '点击或拖拽文件到此区域上传'}
                        </p>
                        <p className="ant-upload-hint text-xs text-muted-foreground">
                            {t('evaluation.upload.hint') || '文件大小不超过 20MB'}
                        </p>
                    </Dragger>

                    {/* 上传要求提示 */}
                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded text-sm">
                        <div className="font-medium mb-2 text-blue-800">
                            {t('evaluation.upload.requirements') || '上传要求：'}
                        </div>
                        <ol className="list-decimal list-inside space-y-1 text-blue-700">
                            <li>{t('evaluation.upload.format-hint') || '建议格式 [{"question":"问题+选项","answer":"正确选项"},{}]'}</li>
                            <li>{t('evaluation.upload.extension-hint') || '必须为 json/jsonl 格式'}</li>
                        </ol>
                    </div>

                    {/* 文件格式错误提示 */}
                    {fileError && (
                        <div className="mt-4 p-4 bg-red-50 border border-red-300 rounded text-red-700 text-sm">
                            <div className="font-medium mb-2">
                                {t('evaluation.upload.validation-error') || '验证失败'}
                            </div>
                            <div>{fileError}</div>
                        </div>
                    )}
                </Form.Item>

                {/* 上传错误提示 */}
                {uploadError && (
                    <div className="mt-4 p-4 bg-red-50 border border-red-300 rounded text-red-700 text-sm">
                        <div className="font-medium mb-2 text-base">
                            Upload Failed
                        </div>
                        <div className="whitespace-pre-line mb-3">
                            {t('evaluation.upload.error.upload-failed-prefix') + '：' + uploadError}
                        </div>
                        <div className="mt-2 pt-2 border-t border-red-200 text-xs text-red-500">
                            {t('evaluation.upload.retry-hint') || '请修改文件后重新上传，或点击取消关闭窗口后重试'}
                        </div>
                    </div>
                )}

                {isUploading && (
                    <div className="mt-4">
                        <Progress
                            percent={uploadProgress}
                            status="active"
                            strokeColor={{ from: '#108ee9', to: '#87d068' }}
                        />
                        <p className="text-xs text-muted-foreground text-center mt-2">
                            {t('evaluation.upload.processing') || '正在上传文件...'}
                        </p>
                    </div>
                )}
            </Form>
        </Modal>
    );
};

export default memo(EvaluationUploadModal);
