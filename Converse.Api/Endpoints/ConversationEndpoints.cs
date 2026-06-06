using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Converse.Api.Stt;
using Converse.Api.Tts;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace Converse.Api.Endpoints;

internal static class ConversationEndpoints
{
    public static IEndpointRouteBuilder MapConversationEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/conversations", async (
            CreateConversationRequest req,
            IConversationStore store,
            IOptions<LlmOptions> opts) =>
        {
            var provider = string.IsNullOrWhiteSpace(req.LlmProvider)
                ? opts.Value.DefaultProvider
                : req.LlmProvider;

            var session = store.Create(req.SystemPrompt, provider);
            return Results.Created($"/conversations/{session.Id}", new { id = session.Id });
        }).Produces(201);

        app.MapPost("/conversations/{id:guid}/turn", async (
            Guid id,
            [FromForm] IFormFile audio,
            ConversationOrchestrator orchestrator,
            ISpeechToTextService stt,
            ITextToSpeechService tts,
            HttpContext httpContext,
            CancellationToken ct) =>
        {
            if (!stt.IsReady)
                return Results.Problem("Whisper STT is not ready; check model path configuration.", statusCode: 503);
            if (!tts.IsReady)
                return Results.Problem("Supertonic TTS is not ready; check model path configuration.", statusCode: 503);

            ConversationTurnResult result;
            try
            {
                await using var stream = audio.OpenReadStream();
                result = await orchestrator.RunTurnAsync(id, stream, ct);
            }
            catch (SessionNotFoundException)
            {
                return Results.NotFound();
            }

            httpContext.Response.Headers.Append("X-User-Transcript",
                Uri.EscapeDataString(result.UserText));
            httpContext.Response.Headers.Append("X-Assistant-Text",
                Uri.EscapeDataString(result.AssistantText));

            return Results.File(result.WavBytes, "audio/wav");
        }).WithMetadata(new RequestSizeLimitAttribute(50 * 1024 * 1024))
          .DisableAntiforgery();

        app.MapGet("/conversations/{id:guid}", (Guid id, IConversationStore store) =>
        {
            if (!store.TryGet(id, out var session))
                return Results.NotFound();

            return Results.Ok(new
            {
                session.Id,
                session.SystemPrompt,
                session.LlmProvider,
                session.CreatedAt,
                Turns = session.Turns.Select(t => new
                {
                    role = t.Role.ToString(),
                    t.Content,
                    t.CreatedAt,
                }),
            });
        });

        app.MapDelete("/conversations/{id:guid}", (Guid id, IConversationStore store) =>
        {
            store.Delete(id);
            return Results.NoContent();
        });

        return app;
    }
}

internal sealed record CreateConversationRequest(string? SystemPrompt, string? LlmProvider);
