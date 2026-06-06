namespace Converse.Api.Conversation;

public sealed class SessionNotFoundException(Guid id)
    : Exception($"Session {id} not found.");
