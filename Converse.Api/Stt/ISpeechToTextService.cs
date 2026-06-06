namespace Converse.Api.Stt;

public interface ISpeechToTextService
{
    // Whether the underlying model loaded successfully (used by /health).
    bool IsReady { get; }

    // Transcribe 16 kHz mono float32 PCM (samples in [-1, 1]) to text.
    // Throws InvalidOperationException if the service isn't ready.
    Task<string> TranscribeAsync(float[] pcm, CancellationToken ct);
}
