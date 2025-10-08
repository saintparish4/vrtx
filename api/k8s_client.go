package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	metricsv "k8s.io/metrics/pkg/client/clientset/versioned"
)

type K8sClient struct {
	clientset     *kubernetes.Clientset
	metricsClient *metricsv.Clientset
}

func NewK8sClient() (*K8sClient, error) {
	config, err := getK8sConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get k8s config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %w", err)
	}

	metricsClient, err := metricsv.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics client: %w", err)
	}

	return &K8sClient{
		clientset:     clientset,
		metricsClient: metricsClient,
	}, nil
}

func getK8sConfig() (*rest.Config, error) {
	// Try in-cluster config first
	config, err := rest.InClusterConfig()
	if err == nil {
		return config, nil
	}

	// Fall back to kubeconfig
	kubeconfigPath := os.Getenv("KUBECONFIG")
	if kubeconfigPath == "" {
		if home := homedir.HomeDir(); home != "" {
			kubeconfigPath = filepath.Join(home, ".kube", "config")
		}
	}

	config, err = clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to build config: %w", err)
	}

	return config, nil
}

func (k *K8sClient) GetNamespaces(ctx context.Context) ([]string, error) {
	namespaces, err := k.clientset.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	names := make([]string, len(namespaces.Items))
	for i, ns := range namespaces.Items {
		names[i] = ns.Name
	}
	return names, nil
}

func (k *K8sClient) GetDeployments(ctx context.Context, namespace string) ([]DeploymentInfo, error) {
	if namespace == "" {
		namespace = "default"
	}

	deployments, err := k.clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	result := make([]DeploymentInfo, len(deployments.Items))
	for i, d := range deployments.Items {
		result[i] = DeploymentInfo{
			Name:      d.Name,
			Namespace: d.Namespace,
			Replicas:  *d.Spec.Replicas,
			Ready:     d.Status.ReadyReplicas,
			Available: d.Status.AvailableReplicas,
		}
	}
	return result, nil
}

func (k *K8sClient) GetPods(ctx context.Context, namespace string) ([]PodInfo, error) {
	if namespace == "" {
		namespace = "default"
	}

	pods, err := k.clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	result := make([]PodInfo, len(pods.Items))
	for i, p := range pods.Items {
		result[i] = PodInfo{
			Name:      p.Name,
			Namespace: p.Namespace,
			Status:    string(p.Status.Phase),
			Node:      p.Spec.NodeName,
		}
	}
	return result, nil
}

func (k *K8sClient) GetResourceMetrics(ctx context.Context, namespace string) (*ResourceMetrics, error) {
	if namespace == "" {
		namespace = "default"
	}

	podMetrics, err := k.metricsClient.MetricsV1beta1().PodMetricses(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var totalCPU, totalMemory int64
	for _, pm := range podMetrics.Items {
		for _, c := range pm.Containers {
			totalCPU += c.Usage.Cpu().MilliValue()
			totalMemory += c.Usage.Memory().Value()
		}
	}

	return &ResourceMetrics{
		CPUMillicores: totalCPU,
		MemoryBytes:   totalMemory,
		PodCount:      len(podMetrics.Items),
	}, nil
}

func (k *K8sClient) ScaleDeployment(ctx context.Context, namespace, name string, replicas int32) error {
	if namespace == "" {
		namespace = "default"
	}

	scale, err := k.clientset.AppsV1().Deployments(namespace).GetScale(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get scale: %w", err)
	}

	scale.Spec.Replicas = replicas

	_, err = k.clientset.AppsV1().Deployments(namespace).UpdateScale(ctx, name, scale, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update scale: %w", err)
	}

	return nil
}
