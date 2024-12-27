package api

import (
	"fmt"
	"sync"

	"github.com/xzeldon/whisper-api-server/pkg/whisper"
)

type WhisperState struct {
	model   *whisper.Model
	context *whisper.IContext
	media   *whisper.IMediaFoundation
	params  *whisper.FullParams
	mutex   sync.Mutex
	closed  bool
}

func InitializeWhisperState(modelPath string, lang int32) (*WhisperState, error) {
	lib, err := whisper.New(whisper.LlDebug, whisper.LfUseStandardError, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize whisper library: %w", err)
	}

	model, err := lib.LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	context, err := model.CreateContext()
	if err != nil {
		model.Release()
		return nil, fmt.Errorf("failed to create context: %w", err)
	}

	media, err := lib.InitMediaFoundation()
	if err != nil {
		context.Release()
		model.Release()
		return nil, fmt.Errorf("failed to init media foundation: %w", err)
	}

	params, err := context.FullDefaultParams(whisper.SsBeamSearch)
	if err != nil {
		media.Release()
		context.Release()
		model.Release()
		return nil, fmt.Errorf("failed to get default params: %w", err)
	}

	params.SetLanguage(lang)

	return &WhisperState{
		model:   model,
		context: context,
		media:   media,
		params:  params,
	}, nil
}

func (ws *WhisperState) ResetContext() error {
	ws.mutex.Lock()
	defer ws.mutex.Unlock()

	if ws.closed {
		return fmt.Errorf("whisper state is already closed")
	}

	// Release the old context
	if ws.context != nil {
		ws.context.Release()
	}

	// Create a new context
	context, err := ws.model.CreateContext()
	if err != nil {
		return fmt.Errorf("failed to create new context: %w", err)
	}

	ws.context = context
	return nil
}

func (ws *WhisperState) Cleanup() error {
	ws.mutex.Lock()
	defer ws.mutex.Unlock()

	if ws.closed {
		return nil
	}

	if ws.params != nil {
		ws.params.Release()
		ws.params = nil
	}

	if ws.media != nil {
		ws.media.Release()
		ws.media = nil
	}

	if ws.context != nil {
		ws.context.Release()
		ws.context = nil
	}

	if ws.model != nil {
		ws.model.Release()
		ws.model = nil
	}

	ws.closed = true
	return nil
}
func getResult(ctx *whisper.IContext) (string, error) {
	results := &whisper.ITranscribeResult{}
	ctx.GetResults(whisper.RfTokens|whisper.RfTimestamps, &results)
	defer results.Release() // Keep this as ITranscribeResult needs to be released

	length, err := results.GetSize()
	if err != nil {
		return "", fmt.Errorf("failed to get results size: %w", err)
	}

	segments := results.GetSegments(length.CountSegments)
	var result string

	for _, seg := range segments {
		result += seg.Text()
		// Removed seg.Release() since segments are value types
	}

	return result, nil
}
