package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type ScalingHandler struct {
	k8s     *K8sClient
	history []ScalingEvent
	mu      sync.RWMutex
}

func NewScalingHandler(k8s *K8sClient) *ScalingHandler {
	return &ScalingHandler{
		k8s:     k8s,
		history: make([]ScalingEvent, 0),
	}
}

func (h *ScalingHandler) ScaleDeployment(c *fiber.Ctx) error {
	var req ScaleRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	if req.Namespace == "" {
		req.Namespace = "default"
	}

	if req.Deployment == "" {
		return c.Status(400).JSON(fiber.Map{
			"error": "Deployment name is required",
		})
	}

	if req.Replicas < 0 {
		return c.Status(400).JSON(fiber.Map{
			"error": "Replicas must be non-negative",
		})
	}

	// Get current state
	deployments, err := h.k8s.GetDeployments(c.Context(), req.Namespace)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error": "Failed to get deployment info",
			"details": err.Error(),
		})
	}

	var currentReplicas int32
	for _, d := range deployments {
		if d.Name == req.Deployment {
			currentReplicas = d.Replicas
			break
		}
	}

	// Perform scaling
	err = h.k8s.ScaleDeployment(c.Context(), req.Namespace, req.Deployment, req.Replicas)
	if err != nil {
		h.recordScalingEvent(req.Namespace, req.Deployment, currentReplicas, req.Replicas, "Manual scaling", false)
		return c.Status(500).JSON(fiber.Map{
			"error": "Failed to scale deployment",
			"details": err.Error(),
		})
	}

	h.recordScalingEvent(req.Namespace, req.Deployment, currentReplicas, req.Replicas, "Manual scaling", true)

	return c.JSON(fiber.Map{
		"success": true,
		"namespace": req.Namespace,
		"deployment": req.Deployment,
		"old_replicas": currentReplicas,
		"new_replicas": req.Replicas,
	})
}

func (h *ScalingHandler) PredictAndScale(c *fiber.Ctx) error {
	var req PredictAndScaleRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{
			"error": "Invalid request body",
			"details": err.Error(),
		})
	}

	if req.Namespace == "" {
		req.Namespace = "default"
	}

	if req.MLEndpoint == "" {
		req.MLEndpoint = "http://localhost:8000" // Default ML service endpoint
	}

	// Get current metrics
	metrics, err := h.k8s.GetResourceMetrics(c.Context(), req.Namespace)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error": "Failed to get current metrics",
			"details": err.Error(),
		})
	}

	// Fetch prediction from ML service
	prediction, err := h.fetchPrediction(req.MLEndpoint, req.Namespace, req.Deployment)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error": "Failed to get prediction",
			"details": err.Error(),
		})
	}

	// Get current deployment info
	deployments, err := h.k8s.GetDeployments(c.Context(), req.Namespace)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error": "Failed to get deployment info",
			"details": err.Error(),
		})
	}

	var currentReplicas int32
	for _, d := range deployments {
		if d.Name == req.Deployment {
			currentReplicas = d.Replicas
			break
		}
	}

	// Calculate scaling decision
	decision := h.calculateScalingDecision(currentReplicas, metrics, prediction)

	// Apply scaling if needed
	if decision.RecommendedReplicas != currentReplicas {
		err = h.k8s.ScaleDeployment(c.Context(), req.Namespace, req.Deployment, decision.RecommendedReplicas)
		if err != nil {
			h.recordScalingEvent(req.Namespace, req.Deployment, currentReplicas, decision.RecommendedReplicas, decision.Reason, false)
			return c.Status(500).JSON(fiber.Map{
				"error": "Failed to apply scaling",
				"details": err.Error(),
			})
		}
		h.recordScalingEvent(req.Namespace, req.Deployment, currentReplicas, decision.RecommendedReplicas, decision.Reason, true)
	}

	return c.JSON(fiber.Map{
		"success":  true,
		"decision": decision,
		"scaled":   decision.RecommendedReplicas != currentReplicas,
	})
}

func (h *ScalingHandler) GetScalingHistory(c *fiber.Ctx) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	limit := c.QueryInt("limit", 50)
	if limit > len(h.history) {
		limit = len(h.history)
	}

	// Return most recent events
	start := len(h.history) - limit
	if start < 0 {
		start = 0
	}

	return c.JSON(fiber.Map{
		"events": h.history[start:],
		"count":  limit,
		"total":  len(h.history),
	})
}

func (h *ScalingHandler) fetchPrediction(endpoint, namespace, deployment string) (*MLPrediction, error) {
	url := fmt.Sprintf("%s/api/predictions?namespace=%s&deployment=%s", endpoint, namespace, deployment)
	
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned %d: %s", resp.StatusCode, string(body))
	}

	var prediction MLPrediction
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		return nil, err
	}

	return &prediction, nil
}

func (h *ScalingHandler) calculateScalingDecision(currentReplicas int32, metrics *ResourceMetrics, prediction *MLPrediction) ScalingDecision {
	// Simple scaling logic based on predicted CPU
	// Scale up if predicted CPU > 70%, scale down if < 30%
	
	currentCPUPerPod := float64(metrics.CPUMillicores) / float64(currentReplicas) / 1000.0
	predictedLoad := prediction.PredictedCPU / currentCPUPerPod
	
	var recommendedReplicas int32
	var reason string

	if predictedLoad > 0.7 && prediction.Confidence > 0.6 {
		// Scale up proactively
		recommendedReplicas = int32(float64(currentReplicas) * 1.5)
		reason = fmt.Sprintf("High predicted load (%.2f%%) with %.0f%% confidence", predictedLoad*100, prediction.Confidence*100)
	} else if predictedLoad < 0.3 && prediction.Confidence > 0.6 {
		// Scale down to save resources
		recommendedReplicas = int32(float64(currentReplicas) * 0.7)
		if recommendedReplicas < 1 {
			recommendedReplicas = 1
		}
		reason = fmt.Sprintf("Low predicted load (%.2f%%) with %.0f%% confidence", predictedLoad*100, prediction.Confidence*100)
	} else {
		// Keep current replicas
		recommendedReplicas = currentReplicas
		reason = "No scaling needed - predicted load within normal range"
	}

	return ScalingDecision{
		CurrentReplicas:     currentReplicas,
		RecommendedReplicas: recommendedReplicas,
		Reason:              reason,
		Prediction:          prediction,
	}
}

func (h *ScalingHandler) recordScalingEvent(namespace, deployment string, oldReplicas, newReplicas int32, reason string, success bool) {
	h.mu.Lock()
	defer h.mu.Unlock()

	event := ScalingEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now(),
		Namespace:   namespace,
		Deployment:  deployment,
		OldReplicas: oldReplicas,
		NewReplicas: newReplicas,
		Reason:      reason,
		Success:     success,
	}

	h.history = append(h.history, event)

	// Keep only last 1000 events to prevent memory issues
	if len(h.history) > 1000 {
		h.history = h.history[len(h.history)-1000:]
	}
}