import { InferenceConfig } from './types';

export interface ValidationError {
  field: string;
  section: keyof InferenceConfig;
  message: string;
  type: 'error' | 'warning';
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
}

// 端口范围验证
export const validatePort = (
  port: number,
  fieldName: string,
  section: keyof InferenceConfig = 'ports',
): ValidationError | null => {
  if (!port || port <= 0) {
    return {
      field: fieldName,
      section,
      message: `${fieldName} 不能为空`,
      type: 'error',
    };
  }
  if (port < 1 || port > 65535) {
    return {
      field: fieldName,
      section,
      message: `${fieldName} 必须在 1-65535 范围内`,
      type: 'error',
    };
  }
  if (port <= 1024) {
    return {
      field: fieldName,
      section,
      message: `${fieldName} (${port}) 为系统保留端口（1-1024），可能需要 root 权限`,
      type: 'warning',
    };
  }
  return null;
};

// IP 地址格式验证
export const validateIP = (ip: string): ValidationError | null => {
  if (!ip || ip.trim() === '') {
    return null; // 允许为空
  }
  const ipv4Regex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
  const hostnameRegex = /^[a-zA-Z0-9][-a-zA-Z0-9]*$/;
  
  if (!ipv4Regex.test(ip) && !hostnameRegex.test(ip) && ip !== 'localhost') {
    return {
      field: 'HOST_IP',
      section: 'env',
      message: 'HOST_IP 格式不正确，应为 IPv4 地址、主机名或 localhost',
      type: 'error',
    };
  }
  return null;
};

// CUDA 设备格式验证
export const validateCUDA = (devices: string): ValidationError | null => {
  if (!devices || devices.trim() === '') {
    return null; // 允许为空
  }
  const cudaRegex = /^[\d,]+$/;
  if (!cudaRegex.test(devices)) {
    return {
      field: 'CUDA_VISIBLE_DEVICES',
      section: 'env',
      message: 'CUDA_VISIBLE_DEVICES 格式不正确，应为逗号分隔的数字（如 0,1,2）',
      type: 'error',
    };
  }
  return null;
};

// GPU 内存利用率验证
export const validateGPUMemory = (util: number): ValidationError | null => {
  if (util === undefined || util === null) {
    return null; // 允许为空
  }
  if (util < 0.1 || util > 1.0) {
    return {
      field: 'GPU_MEMORY_UTILIZATION',
      section: 'runtime',
      message: 'GPU 内存利用率应在 0.1-1.0 范围内',
      type: 'error',
    };
  }
  if (util > 0.95) {
    return {
      field: 'GPU_MEMORY_UTILIZATION',
      section: 'runtime',
      message: `GPU 内存利用率设置较高 (${util})，可能导致 OOM 风险`,
      type: 'warning',
    };
  }
  return null;
};

// Tensor 并行大小验证
export const validateTensorParallel = (size: number): ValidationError | null => {
  if (size === undefined || size === null || size === 0) {
    return null; // 允许为空或 0
  }
  if (size < 1 || size > 8) {
    return {
      field: 'TENSOR_PARALLEL_SIZE',
      section: 'runtime',
      message: 'Tensor Parallel Size 应在 1-8 范围内',
      type: 'error',
    };
  }
  return null;
};

// 必需字段验证
export const validateRequired = (
  value: string | undefined,
  fieldName: string,
  displayName: string
): ValidationError | null => {
  if (!value || value.trim() === '') {
    return {
      field: fieldName,
      section: 'env',
      message: `${displayName} 不能为空`,
      type: 'error',
    };
  }
  return null;
};

// 检查端口重复
// 完整配置验证
export const validateConfig = (config: Partial<InferenceConfig>): ValidationResult => {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  // 验证端口
  if (config.ports) {
    const { ports } = config;
    
    // 单个端口验证
    const portFields = [
      { key: 'VLLM_OPENAI_PORT', value: ports.VLLM_OPENAI_PORT },
    ];

    portFields.forEach(({ key, value }) => {
      const error = validatePort(value || 0, key);
      if (error) {
        if (error.type === 'error') {
          errors.push(error);
        } else {
          warnings.push(error);
        }
      }
    });

  }

  // 验证环境变量
  if (config.env) {
    const { env } = config;

    // IP 验证
    const ipError = validateIP(env.HOST_IP || '');
    if (ipError) {
      if (ipError.type === 'error') {
        errors.push(ipError);
      } else {
        warnings.push(ipError);
      }
    }

    // CUDA 验证
    const cudaError = validateCUDA(env.CUDA_VISIBLE_DEVICES || '');
    if (cudaError) {
      if (cudaError.type === 'error') {
        errors.push(cudaError);
      } else {
        warnings.push(cudaError);
      }
    }

    const masterPortError = validatePort(env.MASTER_PORT || 0, 'MASTER_PORT', 'env');
    if (masterPortError) {
      if (masterPortError.type === 'error') {
        errors.push(masterPortError);
      } else {
        warnings.push(masterPortError);
      }
    }

    if (
      env.MASTER_PORT &&
      config.ports?.VLLM_OPENAI_PORT &&
      env.MASTER_PORT === config.ports.VLLM_OPENAI_PORT
    ) {
      errors.push({
        field: 'MASTER_PORT',
        section: 'env',
        message: 'MASTER_PORT 不能与 VLLM_OPENAI_PORT 重复',
        type: 'error',
      });
    }

    // 必需字段验证
    const modelNameError = validateRequired(env.MODEL_NAME, 'MODEL_NAME', '模型名称');
    if (modelNameError) errors.push(modelNameError);

    const modelPathError = validateRequired(env.MODEL_PATH, 'MODEL_PATH', '模型路径');
    if (modelPathError) errors.push(modelPathError);
  }

  // 验证运行时配置
  if (config.runtime) {
    const { runtime } = config;

    const gpuMemError = validateGPUMemory(runtime.GPU_MEMORY_UTILIZATION || 0);
    if (gpuMemError) {
      if (gpuMemError.type === 'error') {
        errors.push(gpuMemError);
      } else {
        warnings.push(gpuMemError);
      }
    }

    const tensorError = validateTensorParallel(runtime.TENSOR_PARALLEL_SIZE || 0);
    if (tensorError) {
      if (tensorError.type === 'error') {
        errors.push(tensorError);
      } else {
        warnings.push(tensorError);
      }
    }

  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
};
