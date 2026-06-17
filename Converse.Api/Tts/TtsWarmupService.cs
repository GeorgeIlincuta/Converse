using System.Diagnostics;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Converse.Api.Tts;

// Runs one throwaway synthesis at startup (on a background thread) so the first
// real request doesn't pay the model/DirectML shader warm-up cost (~3s on GPU).
// Fire-and-forget: it never blocks startup and never fails the host.
public sealed class TtsWarmupService : IHostedService
{
    private readonly ITextToSpeechService _tts;
    private readonly ILogger<TtsWarmupService> _logger;

    public TtsWarmupService(ITextToSpeechService tts, ILogger<TtsWarmupService> logger)
    {
        _tts = tts;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _ = Task.Run(async () =>
        {
            if (!_tts.IsReady)
            {
                _logger.LogInformation("TTS warm-up skipped: synthesizer not ready.");
                return;
            }

            try
            {
                var sw = Stopwatch.StartNew();
                await _tts.SynthesizeAsync("Aufwärmen.", CancellationToken.None);
                _logger.LogInformation("TTS warm-up complete in {Ms}ms.", sw.ElapsedMilliseconds);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "TTS warm-up failed (non-fatal).");
            }
        }, CancellationToken.None);

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
