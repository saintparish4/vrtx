package main

import (
	"github.com/gofiber/fiber/v2"
)

type ClusterHandler struct {
	k8s *K8sClient
}

func NewClusterHandler(k8s *K8sClient) *ClusterHandler {
	return &ClusterHandler{k8s: k8s}
}

func (h *ClusterHandler) GetNamespaces(c *fiber.Ctx) error {
	namespaces, err := h.k8s.GetNamespaces(c.Context())
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error":   "Failed to get namespaces",
			"details": err.Error(),
		})
	}

	return c.JSON(fiber.Map{
		"namespaces": namespaces,
		"count":      len(namespaces),
	})
}

func (h *ClusterHandler) GetDeployments(c *fiber.Ctx) error {
	namespace := c.Query("namespace", "default")

	deployments, err := h.k8s.GetDeployments(c.Context(), namespace)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error":   "Failed to get deployments",
			"details": err.Error(),
		})
	}

	return c.JSON(fiber.Map{
		"namespace":   namespace,
		"deployments": deployments,
		"count":       len(deployments),
	})
}

func (h *ClusterHandler) GetPods(c *fiber.Ctx) error {
	namespace := c.Query("namespace", "default")

	pods, err := h.k8s.GetPods(c.Context(), namespace)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error":   "Failed to get pods",
			"details": err.Error(),
		})
	}

	return c.JSON(fiber.Map{
		"namespace": namespace,
		"pods":      pods,
		"count":     len(pods),
	})
}

func (h *ClusterHandler) GetResourceUsage(c *fiber.Ctx) error {
	namespace := c.Query("namespace", "default")

	metrics, err := h.k8s.GetResourceMetrics(c.Context(), namespace)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{
			"error":   "Failed to get resource metrics",
			"details": err.Error(),
		})
	}

	return c.JSON(fiber.Map{
		"namespace": namespace,
		"metrics":   metrics,
		"cpu_cores": float64(metrics.CPUMillicores) / 1000.0,
		"memory_gb": float64(metrics.MemoryBytes) / 1024.0 / 1024.0 / 1024.0,
	})
}

func (h *ClusterHandler) GetResources(c *fiber.Ctx) error {
	// The method GetClusterResources does not exist on *K8sClient.
	// You need to implement this method or call an existing method.
	// For now, return a 501 Not Implemented error.
	return c.Status(fiber.StatusNotImplemented).JSON(fiber.Map{
		"error": "GetClusterResources is not implemented on K8sClient",
	})
}
