using Converse.Api.Audio;
using Converse.Api.Tts;

namespace Converse.Api.Endpoints;

internal static class TtsEndpoints
{
    public static IEndpointRouteBuilder MapTtsEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/tts", async (
            TtsRequest req,
            ITextToSpeechService tts,
            IAudioConverter audio,
            CancellationToken ct) =>
        {
            if (!tts.IsReady)
                return Results.Problem("Supertonic TTS is not ready; check model path configuration.", statusCode: 503);

            var samples = await tts.SynthesizeAsync(req.Text, ct);
            var bytes = audio.PcmToWav(samples, tts.SampleRate);
            return Results.File(bytes, "audio/wav");
        });

        return app;
    }
}

internal sealed record TtsRequest(string Text);
