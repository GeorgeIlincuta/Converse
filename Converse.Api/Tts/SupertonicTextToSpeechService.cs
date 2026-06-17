namespace Converse.Api.Tts;

public sealed class SupertonicTextToSpeechService : ITextToSpeechService
{
    private readonly SupertonicPipeline _pipeline;

    public SupertonicTextToSpeechService(SupertonicPipeline pipeline)
    {
        _pipeline = pipeline;
    }

    public bool IsReady => _pipeline.IsReady;

    public int SampleRate => _pipeline.Config?.SampleRate ?? 0;

    public Task<float[]> SynthesizeAsync(string text, CancellationToken ct)
        => SynthesizeAsync(text, null, null, ct);

    public Task<float[]> SynthesizeAsync(string text, string? voice, string? lang, CancellationToken ct)
        => Task.FromResult(_pipeline.Synthesize(text, voice, lang, ct));
}
