using System.Net;
using System.Text.Json;
using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Converse.Api.Llm;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace Converse.Api.Tests;

public class OpenAICompatibleLlmServiceTests
{
    private static readonly string OkJson =
        """{"choices":[{"message":{"role":"assistant","content":"ok"}}]}""";

    private static OpenAICompatibleLlmService BuildService(
        CapturingHandler handler,
        string baseUrl = "http://localhost:1234",
        string model = "test-model",
        string apiKey = "")
    {
        var http = new HttpClient(handler);
        var options = Options.Create(new LlmOptions
        {
            OpenAICompatible = new OpenAICompatibleOptions
            {
                BaseUrl = baseUrl,
                Model = model,
                ApiKey = apiKey,
            },
        });
        return new OpenAICompatibleLlmService(http, options, NullLogger<OpenAICompatibleLlmService>.Instance);
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
    public async Task CompleteAsync_posts_to_chat_completions_endpoint()
    {
        var handler = OkHandler();
        var svc = BuildService(handler, baseUrl: "http://localhost:1234");

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        handler.CapturedRequest!.RequestUri!.ToString()
            .Should().Be("http://localhost:1234/v1/chat/completions");
    }

    [Fact]
    public async Task CompleteAsync_strips_trailing_slash_from_BaseUrl()
    {
        var handler = OkHandler();
        var svc = BuildService(handler, baseUrl: "http://localhost:1234/");

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        handler.CapturedRequest!.RequestUri!.ToString()
            .Should().Be("http://localhost:1234/v1/chat/completions");
    }

    [Fact]
    public async Task CompleteAsync_sets_authorization_when_apikey_present()
    {
        var handler = OkHandler();
        var svc = BuildService(handler, apiKey: "sk-abc");

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        handler.CapturedRequest!.Headers.Authorization!.Scheme.Should().Be("Bearer");
        handler.CapturedRequest!.Headers.Authorization!.Parameter.Should().Be("sk-abc");
    }

    [Fact]
    public async Task CompleteAsync_omits_authorization_when_apikey_empty()
    {
        var handler = OkHandler();
        var svc = BuildService(handler, apiKey: "");

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        handler.CapturedRequest!.Headers.Authorization.Should().BeNull();
    }

    [Fact]
    public async Task CompleteAsync_prepends_system_message_when_systemPrompt_provided()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        await svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hi") },
            "you are helpful",
            CancellationToken.None);

        using var doc = JsonDocument.Parse(handler.CapturedRequestBody!);
        var messages = doc.RootElement.GetProperty("messages");

        messages[0].GetProperty("role").GetString().Should().Be("system");
        messages[0].GetProperty("content").GetString().Should().Be("you are helpful");
        messages[1].GetProperty("role").GetString().Should().Be("user");
    }

    [Fact]
    public async Task CompleteAsync_omits_system_message_when_systemPrompt_null()
    {
        var handler = OkHandler();
        var svc = BuildService(handler);

        await svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hi") },
            null,
            CancellationToken.None);

        using var doc = JsonDocument.Parse(handler.CapturedRequestBody!);
        var messages = doc.RootElement.GetProperty("messages");

        var roles = Enumerable.Range(0, messages.GetArrayLength())
            .Select(i => messages[i].GetProperty("role").GetString())
            .ToArray();

        roles.Should().NotContain("system");
    }

    [Fact]
    public async Task CompleteAsync_includes_model_in_body()
    {
        var handler = OkHandler();
        var svc = BuildService(handler, model: "llama-3.1");

        await svc.CompleteAsync(new[] { new ChatMessage(Role.User, "hello") }, null, CancellationToken.None);

        using var doc = JsonDocument.Parse(handler.CapturedRequestBody!);
        doc.RootElement.GetProperty("model").GetString().Should().Be("llama-3.1");
    }

    [Fact]
    public async Task CompleteAsync_returns_choice_content()
    {
        var json = """{"choices":[{"message":{"role":"assistant","content":"hello world"}}]}""";
        var handler = OkHandler(json);
        var svc = BuildService(handler);

        var result = await svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hi") }, null, CancellationToken.None);

        result.Should().Be("hello world");
    }

    [Fact]
    public async Task CompleteAsync_throws_when_no_choices()
    {
        var json = """{"choices":[]}""";
        var handler = OkHandler(json);
        var svc = BuildService(handler);

        var act = () => svc.CompleteAsync(
            new[] { new ChatMessage(Role.User, "hi") }, null, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*no choice content*");
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
            new[] { new ChatMessage(Role.User, "hi") }, null, CancellationToken.None);

        await act.Should().ThrowAsync<HttpRequestException>();
    }
}
