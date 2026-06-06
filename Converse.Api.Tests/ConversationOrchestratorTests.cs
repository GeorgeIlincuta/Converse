using Converse.Api.Audio;
using Converse.Api.Conversation;
using Converse.Api.Llm;
using Converse.Api.Stt;
using Converse.Api.Tts;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;

namespace Converse.Api.Tests;

// --- Fakes ---

internal sealed class FakeAudioConverter : IAudioConverter
{
    public float[] ToWhisperPcm(Stream audio) => new[] { 0.1f, 0.2f };
    public byte[] PcmToWav(float[] samples, int sampleRate) => new byte[] { 0xFF, 0xFE };
}

internal sealed class FakeStt : ISpeechToTextService
{
    public bool IsReady => true;
    public float[]? ReceivedPcm { get; private set; }
    public string ReturnText { get; set; } = "hello";

    public Task<string> TranscribeAsync(float[] pcm, CancellationToken ct)
    {
        ReceivedPcm = pcm;
        return Task.FromResult(ReturnText);
    }
}

internal sealed class FakeTts : ITextToSpeechService
{
    public bool IsReady => true;
    public int SampleRate => 24000;
    public string? ReceivedText { get; private set; }

    public Task<float[]> SynthesizeAsync(string text, CancellationToken ct)
    {
        ReceivedText = text;
        return Task.FromResult(new[] { 0.5f, 0.6f });
    }
}

internal sealed class FakeLlm : ILlmService
{
    public IReadOnlyList<ChatMessage>? ReceivedMessages { get; private set; }
    public string? ReceivedSystemPrompt { get; private set; }
    public string ReturnText { get; set; } = "assistant reply";

    public Task<string> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        string? systemPrompt,
        CancellationToken ct)
    {
        ReceivedMessages = messages;
        ReceivedSystemPrompt = systemPrompt;
        return Task.FromResult(ReturnText);
    }
}

// --- Tests ---

public class ConversationOrchestratorTests
{
    private static (ConversationOrchestrator orchestrator, InMemoryConversationStore store, FakeStt stt, FakeTts tts, FakeLlm llm)
        Build(string providerKey = "test-provider")
    {
        var store = new InMemoryConversationStore();
        var audio = new FakeAudioConverter();
        var stt = new FakeStt();
        var tts = new FakeTts();
        var llm = new FakeLlm();

        var services = new ServiceCollection();
        services.AddKeyedSingleton<ILlmService>(providerKey, llm);
        var sp = services.BuildServiceProvider();

        var orchestrator = new ConversationOrchestrator(
            store, audio, stt, tts, sp,
            NullLogger<ConversationOrchestrator>.Instance);

        return (orchestrator, store, stt, tts, llm);
    }

    [Fact]
    public async Task RunTurnAsync_throws_SessionNotFoundException_for_missing_session()
    {
        var (orchestrator, _, _, _, _) = Build();
        var missingId = Guid.NewGuid();

        var act = () => orchestrator.RunTurnAsync(missingId, Stream.Null, CancellationToken.None);

        await act.Should().ThrowAsync<SessionNotFoundException>();
    }

    [Fact]
    public async Task RunTurnAsync_appends_user_turn_with_transcript_before_calling_llm()
    {
        var (orchestrator, store, stt, _, llm) = Build();
        stt.ReturnText = "user said this";

        var session = store.Create(null, "test-provider");
        await orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        llm.ReceivedMessages.Should().NotBeNull();
        llm.ReceivedMessages!.Should().ContainSingle(m =>
            m.Role == Role.User && m.Content == "user said this");
    }

    [Fact]
    public async Task RunTurnAsync_appends_assistant_turn_after_llm_returns()
    {
        var (orchestrator, store, _, _, llm) = Build();
        llm.ReturnText = "assistant said this";

        var session = store.Create(null, "test-provider");
        await orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        session.Turns.Should().Contain(t =>
            t.Role == Role.Assistant && t.Content == "assistant said this");
    }

    [Fact]
    public async Task RunTurnAsync_passes_system_prompt_to_llm()
    {
        var (orchestrator, store, _, _, llm) = Build();
        var session = store.Create("be helpful", "test-provider");

        await orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        llm.ReceivedSystemPrompt.Should().Be("be helpful");
    }

    [Fact]
    public async Task RunTurnAsync_skips_system_role_messages_when_building_llm_history()
    {
        var (orchestrator, store, _, _, llm) = Build();
        var session = store.Create(null, "test-provider");
        session.AddTurn(Role.System, "system instruction");
        session.AddTurn(Role.User, "prior user msg");
        session.AddTurn(Role.Assistant, "prior assistant msg");

        await orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        llm.ReceivedMessages.Should().NotBeNull();
        llm.ReceivedMessages!.Should().NotContain(m => m.Role == Role.System);
    }

    [Fact]
    public async Task RunTurnAsync_resolves_llm_by_session_provider()
    {
        var (orchestrator, store, _, _, llm) = Build(providerKey: "test-provider");
        var session = store.Create(null, "test-provider");

        await orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        llm.ReceivedMessages.Should().NotBeNull();
    }

    [Fact]
    public async Task RunTurnAsync_throws_when_provider_not_registered()
    {
        var (orchestrator, store, _, _, _) = Build(providerKey: "test-provider");
        var session = store.Create(null, "nope");

        var act = () => orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*nope*");
    }

    [Fact]
    public async Task RunTurnAsync_returns_synthesized_wav_with_correct_sample_rate()
    {
        var (orchestrator, store, _, tts, _) = Build();
        var session = store.Create(null, "test-provider");

        var result = await orchestrator.RunTurnAsync(session.Id, Stream.Null, CancellationToken.None);

        result.WavBytes.Should().Equal(new byte[] { 0xFF, 0xFE });
        result.SampleRate.Should().Be(tts.SampleRate);
    }
}
