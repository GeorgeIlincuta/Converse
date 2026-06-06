using System.Net;
using System.Text.Json;
using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Converse.Api.Llm;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace Converse.Api.Tests;

public class GeminiLlmServiceTests
{
    private static readonly string OkJson =
        """{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}""";

    private static GeminiLlmService BuildService(CapturingHandler handler, string apiKey = "test-key", string model = "gemini-2.5-flash")
    {
        var http = new HttpClient(handler);
        var options = Options.Create(new LlmOptions
        {
            Gemini = new GeminiOptions { ApiKey = apiKey, Model = model },
        });
        return new GeminiLlmService(http, options, NullLogger<GeminiLlmService>.Instance);
    }

    private static CapturingHandler OkHandler(string json = "") =>
        new()
        {
            Response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(string.IsNullOrEmpty(json) ? OkJson : json),
            },
        };

    [Fact]
    public async Task CompleteAsync_posts_to_correct_endpoint_with_model_and_key()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        handler.CapturedRequest!.RequestUri!.ToString()
            .Should().EndWith("/models/gemini-2.5-flash:generateContent?key=test-key");
    }

    [Fact]
    public async Task CompleteAsync_includes_systemInstruction_when_systemPrompt_provided()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, "you are helpful", CancellationToken.None);

        using var doc = JsonDocument.Parse(handler.CapturedRequestBody!);
        var text = doc.RootElement
            .GetProperty("systemInstruction")
            .GetProperty("parts")[0]
            .GetProperty("text")
            .GetString();

        text.Should().Be("you are helpful");
    }

    [Fact]
    public async Task CompleteAsync_omits_systemInstruction_when_systemPrompt_null_or_whitespace()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        using var doc = JsonDocument.Parse(handler.CapturedRequestBody!);
        doc.RootElement.TryGetProperty("systemInstruction", out _).Should().BeFalse();
    }

    [Fact]
    public async Task CompleteAsync_maps_user_and_assistant_roles_correctly()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        await svc.CompleteAsync(
            new[]
            {
                new ChatMessage(Role.User, "hi"),
                new ChatMessage(Role.Assistant, "hello"),
            },
            null, CancellationToken.None);

        using var doc = JsonDocument.Parse(handler.CapturedRequestBody!);
        var contents = doc.RootElement.GetProperty("contents");

        contents[0].GetProperty("role").GetString().Should().Be("user");
        contents[0].GetProperty("parts")[0].GetProperty("text").GetString().Should().Be("hi");
        contents[1].GetProperty("role").GetString().Should().Be("model");
        contents[1].GetProperty("parts")[0].GetProperty("text").GetString().Should().Be("hello");
    }

    [Fact]
    public async Task CompleteAsync_throws_for_system_role_messages()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        var act = () => svc.CompleteAsync(
            new[] { new ChatMessage(Role.System, "be helpful") },
            null, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*System role*");
    }

    [Fact]
    public async Task CompleteAsync_returns_trimmed_candidate_text()
    {
        var json = """{"candidates":[{"content":{"parts":[{"text":"  hi there  "}]}}]}""";
        var handler = OkHandler(json);
        var svc = BuildService(handler);

        var result = await svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        result.Should().Be("hi there");
    }

    [Fact]
    public async Task CompleteAsync_throws_when_response_has_no_candidates()
    {
        var json = """{"candidates":[]}""";
        var handler = OkHandler(json);
        var svc = BuildService(handler);

        var act = () => svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*no candidate text*");
    }

    [Fact]
    public async Task CompleteAsync_throws_on_non_success_status()
    {
        var handler = new CapturingHandler
        {
            Response = new HttpResponseMessage(HttpStatusCode.BadRequest)
            {
                Content = new StringContent("bad request"),
            },
        };
        var svc = BuildService(handler);

        var act = () => svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        await act.Should().ThrowAsync<HttpRequestException>();
    }
}
