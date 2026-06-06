namespace Converse.Api.Conversation;

public enum Role { System, User, Assistant }

public sealed record Turn(Role Role, string Content, DateTimeOffset CreatedAt);

public sealed class Session
{
    public Guid Id { get; init; }
    public string? SystemPrompt { get; init; }
    public string LlmProvider { get; init; } = "openai-compatible";
    public DateTimeOffset CreatedAt { get; init; }
    public IReadOnlyList<Turn> Turns => _turns.AsReadOnly();

    private readonly List<Turn> _turns = new();

    public void AddTurn(Role role, string content)
    {
        _turns.Add(new Turn(role, content, DateTimeOffset.UtcNow));
    }
}
