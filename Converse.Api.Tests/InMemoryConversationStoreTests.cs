using FluentAssertions;
using Converse.Api.Conversation;

namespace Converse.Api.Tests;

public class InMemoryConversationStoreTests
{
    private readonly InMemoryConversationStore _store = new();

    [Fact]
    public void Create_assigns_id_and_creation_time()
    {
        var before = DateTimeOffset.UtcNow;
        var session = _store.Create(null);
        var after = DateTimeOffset.UtcNow;

        session.Id.Should().NotBe(Guid.Empty);
        session.CreatedAt.Should().BeOnOrAfter(before).And.BeOnOrBefore(after);
    }

    [Fact]
    public void Create_stores_optional_system_prompt()
    {
        var session = _store.Create("You are a helpful assistant.");

        session.SystemPrompt.Should().Be("You are a helpful assistant.");
    }

    [Fact]
    public void TryGet_returns_true_for_known_session()
    {
        var created = _store.Create(null);

        var found = _store.TryGet(created.Id, out var retrieved);

        found.Should().BeTrue();
        retrieved.Should().BeSameAs(created);
    }

    [Fact]
    public void TryGet_returns_false_for_unknown_id()
    {
        var found = _store.TryGet(Guid.NewGuid(), out var retrieved);

        found.Should().BeFalse();
        retrieved.Should().BeNull();
    }

    [Fact]
    public void Delete_removes_session_and_returns_true()
    {
        var session = _store.Create(null);

        var deleted = _store.Delete(session.Id);
        var found = _store.TryGet(session.Id, out _);

        deleted.Should().BeTrue();
        found.Should().BeFalse();
    }

    [Fact]
    public void Delete_returns_false_for_unknown_id()
    {
        var deleted = _store.Delete(Guid.NewGuid());

        deleted.Should().BeFalse();
    }

    [Fact]
    public void AddTurn_appends_in_order()
    {
        var session = _store.Create(null);

        session.AddTurn(Role.User, "hi");
        session.AddTurn(Role.Assistant, "hello");

        session.Turns.Should().HaveCount(2);
        session.Turns[0].Role.Should().Be(Role.User);
        session.Turns[0].Content.Should().Be("hi");
        session.Turns[1].Role.Should().Be(Role.Assistant);
        session.Turns[1].Content.Should().Be("hello");
    }
}
