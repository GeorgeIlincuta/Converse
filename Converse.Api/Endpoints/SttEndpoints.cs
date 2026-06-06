using Converse.Api.Audio;
using Converse.Api.Stt;
using Microsoft.AspNetCore.Mvc;

namespace Converse.Api.Endpoints;

internal static class SttEndpoints
{
    public static IEndpointRouteBuilder MapSttEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/stt", async (
            [FromForm] IFormFile audio,
            IAudioConverter converter,
            ISpeechToTextService stt,
            CancellationToken ct) =>
        {
            await using var stream = audio.OpenReadStream();
            var pcm = converter.ToWhisperPcm(stream);
            var text = await stt.TranscribeAsync(pcm, ct);
            return Results.Ok(new { text });
        }).DisableAntiforgery();

        return app;
    }
}
