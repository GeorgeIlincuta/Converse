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
    {
        return Task.FromResult(_pipeline.Synthesize(text, ct));
    }
}
