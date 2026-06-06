using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Converse.Api.Llm;

public sealed class GeminiLlmService : ILlmService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly HttpClient _http;
    private readonly GeminiOptions _opts;
    private readonly ILogger<GeminiLlmService> _logger;

    public GeminiLlmService(HttpClient http, IOptions<LlmOptions> options, ILogger<GeminiLlmService> logger)
    {
        _http = http;
        _opts = options.Value.Gemini;
        _logger = logger;
    }

    public async Task<string> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        string? systemPrompt,
        CancellationToken ct)
    {
        var contents = messages.Select(m =>
        {
            if (m.Role == Role.System)
                throw new InvalidOperationException(
                    "Gemini history must not contain System role messages; use systemPrompt instead.");
            return new GeminiContent(
                m.Role == Role.User ? "user" : "model",
                new[] { new GeminiPart(m.Content) });
        }).ToArray();

        var body = new GeminiRequest(
            !string.IsNullOrWhiteSpace(systemPrompt)
                ? new GeminiSystemInstruction(new[] { new GeminiPart(systemPrompt) })
                : null,
            contents);

        var url = $"https://generativelanguage.googleapis.com/v1beta/models/" +
                  $"{Uri.EscapeDataString(_opts.Model)}:generateContent" +
                  $"?key={Uri.EscapeDataString(_opts.ApiKey)}";

        using var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = JsonContent.Create(body, options: JsonOptions),
        };

        using var response = await _http.SendAsync(request, ct);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct);
            _logger.LogError("Gemini request failed with {Status}: {Body}",
                response.StatusCode, errorBody.Length > 500 ? errorBody[..500] : errorBody);
            throw new HttpRequestException(
                $"Gemini request failed with status {response.StatusCode}.", null, response.StatusCode);
        }

        var result = await response.Content.ReadFromJsonAsync<GeminiResponse>(JsonOptions, ct);

        var text = result?.Candidates?.FirstOrDefault()?.Content?.Parts?.FirstOrDefault()?.Text;
        if (text is null)
            throw new InvalidOperationException("Gemini returned no candidate text.");

        return text.Trim();
    }

    // --- Private DTOs ---

    private sealed record GeminiRequest(GeminiSystemInstruction? SystemInstruction, GeminiContent[] Contents);

    private sealed record GeminiSystemInstruction(GeminiPart[] Parts);

    private sealed record GeminiContent(string Role, GeminiPart[] Parts);

    private sealed record GeminiPart(string Text);

    private sealed record GeminiResponse(GeminiCandidate[]? Candidates);

    private sealed record GeminiCandidate(GeminiResponseContent? Content);

    private sealed record GeminiResponseContent(GeminiPart[]? Parts);
}
