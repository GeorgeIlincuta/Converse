using Converse.Api.Audio;
using Converse.Api.Llm;
using Converse.Api.Stt;
using Converse.Api.Tts;
using Microsoft.Extensions.Logging;

namespace Converse.Api.Conversation;

public sealed record ConversationTurnResult(
    byte[] WavBytes,
    int SampleRate,
    string UserText,
    string AssistantText);

public sealed class ConversationOrchestrator(
    IConversationStore store,
    IAudioConverter audio,
    ISpeechToTextService stt,
    ITextToSpeechService tts,
    ILlmService llm,
    ILogger<ConversationOrchestrator> logger)
{
    // On TTS failure (e.g., model not yet loaded), the session keeps the appended Assistant turn.
    // Acceptable for v1: the next turn sees the un-synthesised reply in its history. Revisit if real audio playback errors surface.
    public async Task<ConversationTurnResult> RunTurnAsync(
        Guid sessionId,
        Stream audioStream,
        CancellationToken ct)
    {
        if (!store.TryGet(sessionId, out var session))
            throw new SessionNotFoundException(sessionId);

        var pcm = audio.ToWhisperPcm(audioStream);

        var userText = await stt.TranscribeAsync(pcm, ct);
        logger.LogInformation("Transcribed {Length} chars", userText.Length);

        session.AddTurn(Role.User, userText);

        var messages = session.Turns
            .Where(t => t.Role != Role.System)
            .Select(t => new ChatMessage(t.Role, t.Content))
            .ToList();

        var assistantText = await llm.CompleteAsync(messages, session.SystemPrompt, ct);
        logger.LogInformation("LLM responded with {Length} chars", assistantText.Length);

        session.AddTurn(Role.Assistant, assistantText);

        var samples = await tts.SynthesizeAsync(assistantText, ct);
        logger.LogInformation("Synthesized {Samples} samples @ {Rate}Hz", samples.Length, tts.SampleRate);

        var wavBytes = audio.PcmToWav(samples, tts.SampleRate);

        return new ConversationTurnResult(wavBytes, tts.SampleRate, userText, assistantText);
    }
}
