using FluentAssertions;

namespace Converse.Api.Tests;

public class SmokeTests
{
    [Fact]
    public void TestRunner_IsWorking()
    {
        true.Should().BeTrue();
    }
}
