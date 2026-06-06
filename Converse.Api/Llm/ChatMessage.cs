using Converse.Api.Conversation;

namespace Converse.Api.Llm;

public sealed record ChatMessage(Role Role, string Content);
