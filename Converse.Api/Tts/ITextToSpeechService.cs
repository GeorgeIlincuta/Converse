namespace Converse.Api.Tts;

public interface ITextToSpeechService
{
    bool IsReady { get; }
    int SampleRate { get; }                  // model's native output rate, 0 if not ready
    Task<float[]> SynthesizeAsync(string text, CancellationToken ct);
}
