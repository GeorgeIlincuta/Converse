using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Converse.Api.Llm;

public sealed class LmStudioLlmService : ILlmService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly HttpClient _http;
    private readonly LmStudioOptions _opts;
    private readonly ILogger<LmStudioLlmService> _logger;

    public LmStudioLlmService(HttpClient http, IOptions<LlmOptions> options, ILogger<LmStudioLlmService> logger)
    {
        _http = http;
        _opts = options.Value.LmStudio;
        _logger = logger;
    }

    public async Task<string> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        string? systemPrompt,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(_opts.BaseUrl))
            throw new InvalidOperationException(
                "LM Studio LLM is not configured: Llm:LmStudio:BaseUrl is empty.");

        var chatMessages = new List<OaiMessage>();

        if (!string.IsNullOrWhiteSpace(systemPrompt))
            chatMessages.Add(new OaiMessage("system", systemPrompt));

        foreach (var m in messages)
        {
            var role = m.Role switch
            {
                Role.User => "user",
                Role.Assistant => "assistant",
                Role.System => "system",
                _ => "user",
            };
            chatMessages.Add(new OaiMessage(role, m.Content));
        }

        var body = new OaiRequest(_opts.Model, chatMessages.ToArray());
        var url = $"{_opts.BaseUrl.TrimEnd('/')}/v1/chat/completions";

        using var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = JsonContent.Create(body, options: JsonOptions),
        };

        if (!string.IsNullOrEmpty(_opts.ApiKey))
            request.Headers.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _opts.ApiKey);

        using var response = await _http.SendAsync(request, ct);

        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync(ct);
            _logger.LogError("LM Studio request failed with {Status}: {Body}",
                response.StatusCode, errorBody.Length > 500 ? errorBody[..500] : errorBody);
            throw new HttpRequestException(
                $"LM Studio request failed with status {response.StatusCode}.", null, response.StatusCode);
        }

        var result = await response.Content.ReadFromJsonAsync<OaiResponse>(JsonOptions, ct);

        var content = result?.Choices?.FirstOrDefault()?.Message?.Content;
        if (content is null)
            throw new InvalidOperationException("LM Studio endpoint returned no choice content.");

        return content.Trim();
    }

    private sealed record OaiRequest(string Model, OaiMessage[] Messages);

    private sealed record OaiMessage(string Role, string Content);

    private sealed record OaiResponse(OaiChoice[]? Choices);

    private sealed record OaiChoice(OaiMessage? Message);
}
