using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Converse.Api.Configuration;

namespace Converse.Api.Tts;

public sealed class SupertonicTextToSpeechService : ITextToSpeechService, IDisposable
{
    private readonly ILogger<SupertonicTextToSpeechService> _logger;
    private readonly InferenceSession? _session;

    public bool IsReady { get; }

    // TODO: Once the actual Supertonic model card is known, set SampleRate from the output metadata or a known constant.
    public int SampleRate { get; }

    public SupertonicTextToSpeechService(
        IOptions<SupertonicOptions> options,
        ILogger<SupertonicTextToSpeechService> logger)
    {
        _logger = logger;

        var absolutePath = Path.GetFullPath(options.Value.ModelPath);
        try
        {
            _session = new InferenceSession(absolutePath);
            IsReady = true;
            SampleRate = 24000;

            _logger.LogInformation("Supertonic ONNX loaded from {Path}", absolutePath);

            foreach (var kv in _session.InputMetadata)
            {
                var shape = string.Join("x", kv.Value.Dimensions);
                _logger.LogInformation(
                    "Supertonic input '{Name}': type={ElementType}, shape={Shape}",
                    kv.Key,
                    kv.Value.ElementType,
                    shape);
            }

            foreach (var kv in _session.OutputMetadata)
            {
                var shape = string.Join("x", kv.Value.Dimensions);
                _logger.LogInformation(
                    "Supertonic output '{Name}': type={ElementType}, shape={Shape}",
                    kv.Key,
                    kv.Value.ElementType,
                    shape);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(
                "Failed to load Supertonic ONNX model from '{Path}': {Message}",
                absolutePath,
                ex.Message);
            _session = null;
            IsReady = false;
            SampleRate = 0;
        }
    }

    public Task<float[]> SynthesizeAsync(string text, CancellationToken ct)
    {
        if (!IsReady || _session is null)
            throw new InvalidOperationException(
                "Supertonic TTS service is not ready; check model path configuration.");

        throw new NotImplementedException(
            "Supertonic tokenisation and inference are not yet wired. " +
            "Inspect the InputMetadata logged at startup to determine the model's input contract, " +
            "then implement tokenisation, tensor preparation, and audio decoding.");
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
