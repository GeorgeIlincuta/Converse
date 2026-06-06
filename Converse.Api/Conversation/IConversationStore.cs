using System.Diagnostics.CodeAnalysis;

namespace Converse.Api.Conversation;

public interface IConversationStore
{
    Session Create(string? systemPrompt, string llmProvider);
    bool TryGet(Guid id, [MaybeNullWhen(false)] out Session session);
    bool Delete(Guid id);
}
