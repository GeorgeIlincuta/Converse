using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class SupertonicTextProcessorTests
{
    // Identity-ish indexer big enough to cover ASCII + Latin-1 + combining marks.
    private static SupertonicTextProcessor MakeProcessor()
    {
        var table = new int[0x400];
        for (int i = 0; i < table.Length; i++) table[i] = i;
        return new SupertonicTextProcessor(new UnicodeIndexer(table));
    }

    [Fact]
    public void Preprocess_wraps_with_language_tags_and_appends_period()
    {
        MakeProcessor().Preprocess("Hallo", "de").Should().Be("<de>Hallo.</de>");
    }

    [Fact]
    public void Preprocess_does_not_append_period_when_already_punctuated()
    {
        MakeProcessor().Preprocess("Wie geht es?", "de").Should().Be("<de>Wie geht es?</de>");
    }

    [Fact]
    public void Preprocess_decomposes_umlauts_via_nfkd()
    {
        var result = MakeProcessor().Preprocess("schön", "de");
        result.Should().NotContain("ö");   // precomposed o-umlaut removed by NFKD
        result.Should().Contain("̈");      // combining diaeresis is present
    }

    [Fact]
    public void Preprocess_throws_on_invalid_language()
    {
        var act = () => MakeProcessor().Preprocess("hi", "xx");
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Encode_produces_ids_and_all_ones_mask_of_processed_length()
    {
        var p = MakeProcessor();
        var processed = p.Preprocess("Hallo", "de");
        var (ids, mask) = p.Encode("Hallo", "de");

        ids.Should().HaveCount(processed.Length);
        mask.Should().HaveCount(processed.Length);
        mask.Should().OnlyContain(x => x == 1f);
    }
}
