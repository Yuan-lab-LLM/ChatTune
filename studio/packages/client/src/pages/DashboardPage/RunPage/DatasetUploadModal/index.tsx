import { memo, useState, useCallback } from 'react';
import { Alert, Modal, Form, Input, Tabs, Upload, Progress, message } from 'antd';
import { useTranslation } from 'react-i18next';
import { InboxOutlined, FileOutlined } from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload/interface';
import { useEnvironmentConfig } from '@/hooks/useEnvironmentConfig';

type DatasetType = 'raw' | 'sft' | 'dpo' | 'pt';

interface Props {
    open: boolean;
    onCancel: () => void;
    onUpload: (params: {
        containerName: string;
        datasetType: DatasetType;
        datasetName: string;
        file: File;
    }) => Promise<void>;
    isUploading: boolean;
    uploadProgress: number;
}

const { Dragger } = Upload;

const ALLOWED_EXTENSIONS = ['.tar', '.tar.gz', '.tgz'];
const MAX_SIZE = 20 * 1024 * 1024; // 20MB
const MAX_SIZE_MB = MAX_SIZE / 1024 / 1024;

const formatFileSize = (size = 0) => {
    if (size < 1024 * 1024) {
        return `${(size / 1024).toFixed(1)} KB`;
    }
    return `${(size / 1024 / 1024).toFixed(2)} MB`;
};

const getDatasetNameFromFilename = (filename: string) =>
    filename.replace(/\.(tar\.gz|tgz|tar)$/i, '').replace(/[^\w-]/g, '_');

const DatasetUploadModal = ({ open, onCancel, onUpload, isUploading, uploadProgress }: Props) => {
    const { t } = useTranslation();
    const [form] = Form.useForm();
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [datasetType, setDatasetType] = useState<DatasetType>('raw');
    const [errorMsg, setErrorMsg] = useState<string>('');
    const { defaultContainerName } = useEnvironmentConfig();

    const handleTypeChange = (key: string) => {
        setDatasetType(key as DatasetType);
        form.setFieldsValue({ datasetType: key });
    };

    const validateFile = (file: File): string => {
        // 检查文件大小
        if (file.size > MAX_SIZE) {
            return t('dataset.upload.file-too-large') || `文件大小超过 ${MAX_SIZE_MB}MB 限制`;
        }

        // 检查文件扩展名
        const fileName = file.name.toLowerCase();
        const isAllowed = ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
        if (!isAllowed) {
            return (
                t('dataset.upload.invalid-file-type') ||
                `不支持的文件格式，请上传 ${ALLOWED_EXTENSIONS.join(', ')} 文件`
            );
        }

        return '';
    };

    const beforeUpload: UploadProps['beforeUpload'] = (file) => {
        const validationError = validateFile(file);
        if (validationError) {
            setErrorMsg(validationError);
            message.error(validationError);
            return Upload.LIST_IGNORE;
        }

        setErrorMsg('');
        return false; // 阻止自动上传，改为手动上传
    };

    const handleUploadChange: UploadProps['onChange'] = ({ fileList: newFileList }) => {
        // 只保留最后一个文件
        if (newFileList.length > 1) {
            newFileList = [newFileList[newFileList.length - 1]];
        }
        setFileList(newFileList);

        // 自动填充数据集名称（去掉扩展名）
        if (newFileList.length > 0 && newFileList[0].originFileObj) {
            const file = newFileList[0].originFileObj;
            const nameWithoutExt = getDatasetNameFromFilename(file.name);
            form.setFieldsValue({ datasetName: nameWithoutExt });
            setErrorMsg('');
        }
    };

    const handleSubmit = useCallback(async () => {
        try {
            setErrorMsg('');
            const values = await form.validateFields();

            if (fileList.length === 0 || !fileList[0].originFileObj) {
                message.error(t('dataset.upload.no-file') || '请先选择文件');
                return;
            }

            const file = fileList[0].originFileObj;
            const validationError = validateFile(file);
            if (validationError) {
                setErrorMsg(validationError);
                message.error(validationError);
                return;
            }

            await onUpload({
                containerName: defaultContainerName,
                datasetType: values.datasetType || datasetType,
                datasetName: values.datasetName,
                file,
            });

            // 成功后重置表单
            form.resetFields();
            setFileList([]);
            setErrorMsg('');
        } catch (error: any) {
            // 表单验证失败或上传失败
            console.error('Upload failed:', error);
            if (error && error.message) {
                setErrorMsg(error.message);
            } else {
                setErrorMsg('上传失败，请重试');
            }
        }
    }, [defaultContainerName, form, fileList, onUpload, datasetType, t]);

    const handleCancel = useCallback(() => {
        if (!isUploading) {
            form.resetFields();
            setFileList([]);
            setErrorMsg('');
            onCancel();
        }
    }, [isUploading, form, onCancel]);

    const getFileTypeDescription = () => {
        if (fileList.length > 0 && fileList[0].originFileObj) {
            const file = fileList[0].originFileObj;
            const fileName = file.name.toLowerCase();

            if (fileName.endsWith('.tar') || fileName.endsWith('.tar.gz') || fileName.endsWith('.tgz')) {
                return t('dataset.upload.tar-archive') || 'Tar 归档文件（将自动解压）';
            }
        }
        return '';
    };

    const items = [
        {
            key: 'raw',
            label: t('dataset.type.raw') || '原始数据',
        },
        {
            key: 'sft',
            label: t('dataset.type.sft') || 'SFT 数据',
        },
        {
            key: 'dpo',
            label: t('dataset.type.dpo') || 'DPO 数据',
        },
        {
            key: 'pt',
            label: t('dataset.type.pt') || '预训练文本数据',
        },
    ];

    return (
        <Modal
            title={t('dataset.upload.title') || '上传数据集'}
            open={open}
            onOk={handleSubmit}
            onCancel={handleCancel}
            okText={t('dataset.upload.confirm') || '上传'}
            cancelText={t('common-cancel') || '取消'}
            confirmLoading={isUploading}
            okButtonProps={{ disabled: fileList.length === 0 || isUploading }}
            maskClosable={!isUploading}
            closable={!isUploading}
            width={560}
            className="upload-modal"
        >
            <Form
                form={form}
                layout="vertical"
                initialValues={{
                    datasetType: 'raw',
                }}
            >
                {/* 数据集类型 */}
                <Form.Item
                    name="datasetType"
                    label={t('dataset.type.label') || '数据集类型'}
                    rules={[{ required: true }]}
                >
                    <Tabs
                        items={items}
                        activeKey={datasetType}
                        onChange={handleTypeChange}
                        disabled={isUploading}
                    />
                </Form.Item>

                {/* 数据集名称 */}
                <Form.Item
                    name="datasetName"
                    label={t('dataset.name') || '数据集名称'}
                    rules={[
                        { required: true, message: t('dataset.name-required') || '请输入数据集名称' },
                        { pattern: /^[a-zA-Z0-9_-]+$/, message: t('dataset.name-invalid') || '只能包含字母、数字、下划线和横线' },
                    ]}
                    extra={t('dataset.name-hint') || '将作为文件夹名称，不能包含空格和特殊字符'}
                >
                    <Input placeholder="20260325" disabled={isUploading} />
                </Form.Item>

                {/* 文件上传 */}
                <Form.Item
                    label={t('dataset.upload.file') || '数据文件'}
                    required
                >
                    <Dragger
                        name="file"
                        multiple={false}
                        accept={ALLOWED_EXTENSIONS.join(',')}
                        fileList={fileList}
                        beforeUpload={beforeUpload}
                        onChange={handleUploadChange}
                        onRemove={() => {
                            setFileList([]);
                            setErrorMsg('');
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
                            {t('dataset.upload.drag-hint') || '点击或拖拽文件到此区域上传'}
                        </p>
                        <p className="ant-upload-hint">
                            {t('dataset.upload.file-hint') ||
                                `支持 ${ALLOWED_EXTENSIONS.join(', ')}，单个文件不超过 ${MAX_SIZE_MB}MB`}
                        </p>
                    </Dragger>

                    <Alert
                        className="mt-3"
                        type="info"
                        showIcon
                        message={t('dataset.upload.naming-tip') || '数据集名称会自动按文件名填充，可在上传前修改。'}
                    />

                    {/* 文件格式提示 */}
                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded text-sm">
                        <div className="font-medium mb-2 text-blue-800">{t('dataset.upload.requirements-title') || '上传要求：'}</div>
                        <ul className="list-disc list-inside space-y-1 text-blue-700 mb-3">
                            <li>{t('dataset.upload.requirement-compressed') || '上传的文件为压缩格式（.tar 或 .tar.gz）'}</li>
                            <li>{t('dataset.upload.requirement-json-only') || '解压后应得到一个文件夹，该文件夹内只能包含 .json 文件'}</li>
                            {datasetType !== 'raw' && (
                                <li>{t('dataset.upload.requirement-dataset-info') || '数据文件夹内应存在 dataset_info.json 文件，存放数据集描述'}</li>
                            )}
                        </ul>
                        <div className="font-medium mb-1 text-blue-800">{t('dataset.upload.example-title') || '示例如下：'}</div>
                        <pre className="bg-white p-2 rounded border border-blue-100 text-xs text-gray-600 font-mono">
./20260325{'\n'}
├── data1.json{'\n'}
{datasetType === 'raw' ? '└── data2.json' : '├── data2.json\n└── dataset_info.json'}
                        </pre>
                    </div>
                </Form.Item>

                {/* 文件类型提示 */}
                {fileList.length > 0 && (
                    <div className="mb-4 text-sm text-muted-foreground">
                        <FileOutlined className="mr-2" />
                        {fileList[0].name} ({formatFileSize(fileList[0].size)})
                        <br />
                        <span className="text-xs">{getFileTypeDescription()}</span>
                    </div>
                )}

                {/* 错误消息 */}
                {errorMsg && (
                    <div className="mb-4 p-4 bg-red-50 border border-red-300 rounded text-red-700 text-sm">
                        <div className="font-medium mb-2 text-base">
                            {t('dataset.upload.upload-failed') || '上传失败'}
                        </div>
                        <div className="whitespace-pre-line mb-3">
                            {errorMsg}
                        </div>
                        <div className="mt-2 pt-2 border-t border-red-200 text-xs text-red-500">
                            {t('dataset.upload.retry-hint') || '请修改文件后重新上传，或点击取消关闭窗口后重试'}
                        </div>
                    </div>
                )}

                {/* 上传进度 */}
                {isUploading && (
                    <div className="mb-4">
                        <Progress percent={uploadProgress} status="active" />
                        <p className="text-xs text-muted-foreground mt-1">
                            {t('dataset.upload.uploading') || '正在上传...'}
                        </p>
                    </div>
                )}
            </Form>
        </Modal>
    );
};

export default memo(DatasetUploadModal);
