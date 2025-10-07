package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover" 
)

func main() {
	app := fiber.New(fiber.Config{
		AppName: "K8s Cluster Manager",
		ReadTimeout: 10 * time.Second,
		WriteTimeout: 10 * time.Second, 
	})

	// Middleware
	app.Use(recover.New())
	app.Use(logger.New())
	app.User(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET, POST, PUT, DELETE, OPTIONS",
		AllowHeaders: "Origin, Content-Type, Accept, Authorization", 
	}))

	// Initialize K8s Client
	k8sClient, err := NewK8sClient()
	if err != nil {
		log.Fataf("Failed to initialize K8s client: %v", err)
	}

	// Initialize Handlers
	clusterHandler := NewClusterHandler(k8sClient)
	scalingHandler := NewScalingHandler(k8sClient)

	// Health Check
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "healthy", "timestamp": time.Now()})
	})

	// Cluster info routes
	app.Get("/api/cluster/namespaces", clusterHandler.GetNamespaces)
	app.Get("/api/cluster/deployments", clusterHandler.GetDeployments)
	app.Get("/api/cluster/pods", clusterHandler.GetPods)
	app.Get("/api/cluster/resources", clusterHandler.GetResources)

	// Scaling routes
	app.Post("/api/scaling/scale", scalingHandler.ScaleDeployment)
	app.Post("/api/scaling/predict-and-scale", scalingHandler.PredictAndScale)
	app.Get("/api/scaling/history", scalingHandler.GetScalingHistory)

	// Graceful shutdown
	go func() {
		if err := app.Listen(":8080"); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Handle shutdown signals
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := app.ShutdownWithContext(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited") 
}