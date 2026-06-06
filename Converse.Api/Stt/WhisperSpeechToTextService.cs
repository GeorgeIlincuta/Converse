using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Whisper.net;
using Converse.Api.Configuration;

namespace Converse.Api.Stt;

public sealed class WhisperSpeechToTextService : ISpeechToTextService, IDisposable
{
    private readonly WhisperOptions _options;
    private readonly ILogger<WhisperSpeechToTextService> _logger;
    private readonly WhisperFactory? _factory;

    public bool IsReady { get; }

    public WhisperSpeechToTextService(
        IOptions<WhisperOptions> options,
        ILogger<WhisperSpeechToTextService> logger)
    {
        _options = options.Value;
        _logger = logger;

        var absolutePath = Path.GetFullPath(_options.ModelPath);
        try
        {
            _factory = WhisperFactory.FromPath(absolutePath);
            IsReady = true;
            _logger.LogInformation("Whisper model loaded from {Path}", absolutePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                "Failed to load Whisper model from '{Path}': {Message}",
                absolutePath,
                ex.Message);
            _factory = null;
            IsReady = false;
        }
    }

    public async Task<string> TranscribeAsync(float[] pcm, CancellationToken ct)
    {
        if (!IsReady || _factory is null)
            throw new InvalidOperationException(
                "Whisper STT service is not ready; check model path configuration.");

        await using var processor = _factory
            .CreateBuilder()
            .WithLanguage(_options.Language)
            .Build();

        var segments = new List<string>();
        await foreach (var segment in processor.ProcessAsync(pcm, ct))
        {
            segments.Add(segment.Text);
        }

        return string.Join(" ", segments).Trim();
    }

    public void Dispose()
    {
        _factory?.Dispose();
    }
}
