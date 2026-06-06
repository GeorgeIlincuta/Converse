using FluentAssertions;
using Converse.Api.Audio;

namespace Converse.Api.Tests;

public class NAudioConverterTests
{
    private readonly NAudioConverter _converter = new();

    [Fact]
    public void PcmToWav_starts_with_RIFF_WAVE_header()
    {
        var bytes = _converter.PcmToWav(new float[] { 0f }, 16000);

        var riff = System.Text.Encoding.ASCII.GetString(bytes, 0, 4);
        var wave = System.Text.Encoding.ASCII.GetString(bytes, 8, 4);

        riff.Should().Be("RIFF");
        wave.Should().Be("WAVE");
    }

    [Theory]
    [InlineData(16000)]
    [InlineData(22050)]
    public void PcmToWav_sets_correct_sample_rate_in_header(int sampleRate)
    {
        var bytes = _converter.PcmToWav(new float[] { 0f }, sampleRate);

        var storedRate = BitConverter.ToUInt32(bytes, 24);

        storedRate.Should().Be((uint)sampleRate);
    }

    [Fact]
    public void PcmToWav_data_size_equals_sample_count_times_two()
    {
        var samples = new float[] { 0f, 0.5f, -0.5f, 1f };
        var bytes = _converter.PcmToWav(samples, 16000);

        var dataSize = BitConverter.ToUInt32(bytes, 40);

        dataSize.Should().Be((uint)(samples.Length * 2));
    }

    [Fact]
    public void PcmToWav_clamps_samples_outside_minus_one_to_one()
    {
        var positive = _converter.PcmToWav(new float[] { 2.0f }, 16000);
        var negative = _converter.PcmToWav(new float[] { -2.0f }, 16000);

        var positiveShort = BitConverter.ToInt16(positive, 44);
        var negativeShort = BitConverter.ToInt16(negative, 44);

        positiveShort.Should().Be(short.MaxValue);
        negativeShort.Should().Be(-32767);
    }

    [Fact]
    public void PcmToWav_handles_empty_samples()
    {
        var bytes = _converter.PcmToWav(Array.Empty<float>(), 16000);

        bytes.Should().HaveCount(44);
        var dataSize = BitConverter.ToUInt32(bytes, 40);
        dataSize.Should().Be(0u);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void PcmToWav_throws_for_nonpositive_sample_rate(int sampleRate)
    {
        var act = () => _converter.PcmToWav(new float[] { 0f }, sampleRate);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
