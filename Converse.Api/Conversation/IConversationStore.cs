using System.Diagnostics.CodeAnalysis;

namespace Converse.Api.Conversation;

public interface IConversationStore
{
    Session Create(string? systemPrompt);
    bool TryGet(Guid id, [MaybeNullWhen(false)] out Session session);
    bool Delete(Guid id);
}
