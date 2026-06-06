namespace Converse.Api.Llm;

public interface ILlmService
{
    Task<string> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        string? systemPrompt,
        CancellationToken ct);
}
