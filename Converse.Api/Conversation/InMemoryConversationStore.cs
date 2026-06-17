using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;

namespace Converse.Api.Conversation;

public sealed class InMemoryConversationStore : IConversationStore
{
    private readonly ConcurrentDictionary<Guid, Session> _sessions = new();

    public Session Create(string? systemPrompt)
    {
        var session = new Session
        {
            Id = Guid.NewGuid(),
            SystemPrompt = systemPrompt,
            CreatedAt = DateTimeOffset.UtcNow
        };
        _sessions[session.Id] = session;
        return session;
    }

    public bool TryGet(Guid id, [MaybeNullWhen(false)] out Session session)
    {
        return _sessions.TryGetValue(id, out session);
    }

    public bool Delete(Guid id)
    {
        return _sessions.TryRemove(id, out _);
    }
}
