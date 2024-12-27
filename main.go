package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/xzeldon/whisper-api-server/internal/api"
	"github.com/xzeldon/whisper-api-server/internal/resources"
)

const (
	defaultModelType      = "ggml-large.bin"
	defaultWhisperVersion = "1.12.0"
	shutdownTimeout       = 10 * time.Second
)

func changeWorkingDirectory(e *echo.Echo) error {
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("error getting executable path: %w", err)
	}

	exeDir := filepath.Dir(exePath)
	if err := os.Chdir(exeDir); err != nil {
		return fmt.Errorf("error changing working directory: %w", err)
	}

	cwd, _ := os.Getwd()
	e.Logger.Printf("Current working directory: %s", cwd)
	return nil
}

func main() {
	e := echo.New()
	e.HideBanner = true

	if err := changeWorkingDirectory(e); err != nil {
		e.Logger.Error(err)
		return
	}

	args, err := resources.ParseFlags()
	if err != nil {
		e.Logger.Errorf("Error parsing flags: %v", err)
		return
	}

	if _, err := resources.HandleWhisperDll(defaultWhisperVersion); err != nil {
		e.Logger.Errorf("Error handling Whisper.dll: %v", err)
		return
	}

	if _, err := resources.HandleDefaultModel(defaultModelType); err != nil {
		e.Logger.Errorf("Error handling model file: %v", err)
		return
	}

	e.Use(middleware.CORS())

	whisperState, err := api.InitializeWhisperState(args.ModelPath, args.Language)
	if err != nil {
		e.Logger.Errorf("Error initializing Whisper state: %v", err)
		return
	}

	// Ensure cleanup of whisperState
	defer func() {

		if err := whisperState.Cleanup(); err != nil {
			e.Logger.Errorf("Error during cleanup: %v", err)
		}
	}()
	// Setup graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	// Define routes
	e.POST("/v1/audio/transcriptions", func(c echo.Context) error {
		if err := whisperState.ResetContext(); err != nil {
			e.Logger.Errorf("Failed to reset context: %v", err)
			return c.JSON(http.StatusInternalServerError, map[string]string{
				"error": "Failed to reset transcription context",
			})
		}

		return api.Transcribe(c, whisperState)
	})

	// Start server
	go func() {
		address := fmt.Sprintf("127.0.0.1:%d", args.Port)
		if err := e.Start(address); err != nil && err != http.ErrServerClosed {
			e.Logger.Errorf("Error starting server: %v", err)
		}
	}()

	// Wait for interrupt signal
	<-quit

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	if err := e.Shutdown(ctx); err != nil {
		e.Logger.Errorf("Error during server shutdown: %v", err)
	}
}
