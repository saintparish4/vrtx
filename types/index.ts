export interface MetricData {
  timestamp: number;
  value: number;
  metric_name: string;
  resource_id: string;
}

export interface TrafficPrediction {
  timestamp: number;
  predicted_load: number;
  confidence: number;
  spike_probability: number;
}

export interface ResourceConfig {
  id: string;
  type: 'ec2' | 'gce' | 'azure_vm' | 'kubernetes';
  provider: 'aws' | 'gcp' | 'azure' | 'k8s';
  min_instances: number;
  max_instances: number;
  current_instances: number;
  target_cpu: number;
  target_memory: number;
  cost_per_hour: number;
}

export interface ScalingDecision {
  resource_id: string;
  action: 'scale_up' | 'scale_down' | 'maintain';
  target_instances: number;
  reason: string;
  confidence: number;
  estimated_cost_impact: number;
  timestamp: number;
}

export interface FailurePrediction {
  resource_id: string;
  failure_probability: number;
  predicted_failure_time?: number;
  failure_type: 'cpu_exhaustion' | 'memory_leak' | 'disk_full' | 'network_congestion';
  recommended_action: string;
}

export interface CostOptimization {
  current_cost_per_hour: number;
  optimized_cost_per_hour: number;
  savings_percentage: number;
  recommendations: string[];
}

export interface CloudProvider {
  name: string;
  scale_up: (resourceId: string, targetInstances: number) => Promise<boolean>;
  scale_down: (resourceId: string, targetInstances: number) => Promise<boolean>;
  get_metrics: (resourceId: string, timeRange: number) => Promise<MetricData[]>;
  get_cost_info: (resourceId: string) => Promise<number>;
}

export interface ModelConfig {
  name: string;
  type: 'prophet' | 'lstm' | 'arima' | 'ensemble';
  parameters: Record<string, any>;
  training_window_hours: number;
  prediction_horizon_hours: number;
}
