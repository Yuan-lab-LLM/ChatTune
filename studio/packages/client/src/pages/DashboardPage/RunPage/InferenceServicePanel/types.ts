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
    MODEL_PATH?: string;
    START_SCRIPT?: string;
    LOG_DIR?: string;
    TEST_DIR?: string;
    BENCHMARK_DIR?: string;
    GENERAL_BENCHMARK_DIR?: string;
  };
  runtime: {
    TENSOR_PARALLEL_SIZE?: number;
    GPU_MEMORY_UTILIZATION?: number;
    MAX_TOKENS?: number;
  };
}

// 服务状态数据结构
export interface ServiceStatus {
  name: string;
  port: number;
  status: 'running' | 'stopped';
  rawStatus?: string;
  description?: string;
}
