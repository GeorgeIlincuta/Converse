using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class UnicodeIndexerTests
{
    [Fact]
    public void MapChar_returns_table_value_for_in_range_codepoints()
    {
        var indexer = new UnicodeIndexer(new[] { 5, 7, 9 });
        indexer.MapChar(0).Should().Be(5L);
        indexer.MapChar(2).Should().Be(9L);
    }

    [Fact]
    public void MapChar_returns_zero_for_out_of_range_codepoints()
    {
        var indexer = new UnicodeIndexer(new[] { 5, 7, 9 });
        indexer.MapChar(3).Should().Be(0L);
        indexer.MapChar(-1).Should().Be(0L);
    }
}
