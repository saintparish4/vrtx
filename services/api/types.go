package main

import "time"

type DeploymentInfo struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Replicas  int32  `json:"replicas"`
	Ready     int32  `json:"ready"`
	Available int32  `json:"available"`
}

type PodInfo struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Status    string `json:"status"`
	Node      string `json:"node"`
}

type ResourceMetrics struct {
	CPUMillicores int64 `json:"cpu_millicores"`
	MemoryBytes   int64 `json:"memory_bytes"`
	PodCount      int   `json:"pod_count"`
}

type ScaleRequest struct {
	Namespace  string `json:"namespace"`
	Deployment string `json:"deployment"`
	Replicas   int32  `json:"replicas"`
}

type PredictAndScaleRequest struct {
	Namespace  string `json:"namespace"`
	Deployment string `json:"deployment"`
	MLEndpoint string `json:"ml_endpoint"` // URL of the ML Service
}

type MLPrediction struct {
	PredictedCPU    float64   `json:"predicted_cpu"`
	PredictedMemory float64   `json:"predicted_memory"`
	Confidence      float64   `json:"confidence"`
	Timestamp       time.Time `json:"timestamp"`
}

type ScalingDecision struct {
	CurrentReplicas     int32         `json:"current_replicas"`
	RecommendedReplicas int32         `json:"recommended_replicas"`
	Reason              string        `json:"reason"`
	Prediction          *MLPrediction `json:"prediction,omitempty"`
}

type ScalingEvent struct {
	ID          string    `json:"id"`
	Timestamp   time.Time `json:"timestamp"`
	Namespace   string    `json:"namespace"`
	Deployment  string    `json:"deployment"`
	OldReplicas int32     `json:"old_replicas"`
	NewReplicas int32     `json:"new_replicas"`
	Reason      string    `json:"reason"`
	Success     bool      `json:"success"`
}