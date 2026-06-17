using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class SupertonicHelpersTests
{
    [Fact]
    public void ComputeLatentLen_ceils_to_chunk_size()
    {
        // chunkSize = 512 * 6 = 3072; wavLenMax = 1.0 * 44100 = 44100; ceil(44100/3072) = 15
        SupertonicHelpers.ComputeLatentLen(1.0f, 44100, 512, 6).Should().Be(15);
    }

    [Fact]
    public void LatentLengthFor_ceils_wav_length()
    {
        // latentSize = 3072; ceil(44100/3072) = 15
        SupertonicHelpers.LatentLengthFor(44100, 512, 6).Should().Be(15);
    }

    [Fact]
    public void Mask_is_ones_then_zeros()
    {
        var mask = SupertonicHelpers.Mask(length: 3, total: 5);
        mask.Should().Equal(1f, 1f, 1f, 0f, 0f);
    }

    [Fact]
    public void ChunkText_returns_single_chunk_for_short_text()
    {
        var chunks = SupertonicHelpers.ChunkText("Hallo, wie geht es dir?", 300);
        chunks.Should().ContainSingle();
    }

    [Fact]
    public void ChunkText_splits_long_text_into_multiple_chunks()
    {
        var sentence = "Dies ist ein deutscher Satz. ";
        var text = string.Concat(Enumerable.Repeat(sentence, 40)); // > 300 chars
        var chunks = SupertonicHelpers.ChunkText(text, 100);
        chunks.Count.Should().BeGreaterThan(1);
    }
}
