// 推理服务配置数据结构
export interface InferenceConfig {
  ports: {
    VLLM_OPENAI_PORT?: number;
    INFERENCE_PORT?: number;
    UI_PORT?: number;
    DATA_ANNOTATION_PORT?: number;
  };
  env: {
    HOST_IP?: string;
    CUDA_VISIBLE_DEVICES?: string;
    MODEL_NAME?: string;
    MODEL_PARAM_B?: string;
    MODEL_PATH?: string;
    START_SCRIPT?: string;
    LOG_DIR?: string;
    TEST_DIR?: string;
    BENCHMARK_DIR?: string;
    GENERAL_BENCHMARK_DIR?: string;
    MASTER_PORT?: number;
  };
  runtime: {
    TENSOR_PARALLEL_SIZE?: number;
    GPU_MEMORY_UTILIZATION?: number;
    GPU_UTILIZATION_THRESHOLD?: number;
    MAX_TOKENS?: number;
  };
}

// 服务状态数据结构
export interface ServiceStatus {
  name: string;
  port: number;
  status: 'running' | 'stopped' | 'starting' | 'failed' | 'degraded';
  rawStatus?: string;
  description?: string;
  serviceKey?: string;
  displayName?: string;
  instanceId?: string;
  node?: string;
  gpus?: string[];
  model?: string;
  ports?: Record<string, number | string>;
  reservationId?: string;
  ownerUserId?: string;
  startedAt?: string;
  logDir?: string;
  isInstance?: boolean;
  isPortStatus?: boolean;
}

