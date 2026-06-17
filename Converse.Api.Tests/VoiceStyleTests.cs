using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class VoiceStyleTests
{
    private const string Json = """
    {
      "style_ttl": { "dims": [1, 1, 2], "data": [[[0.5, -0.5]]], "type": "float32" },
      "style_dp":  { "dims": [1, 1, 3], "data": [[[1.0, 2.0, 3.0]]], "type": "float32" },
      "metadata":  { "source_file": "x" }
    }
    """;

    [Fact]
    public void Load_parses_shapes_and_flattened_data()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, Json);
        try
        {
            var style = VoiceStyle.Load(path);
            style.TtlShape.Should().Equal(1L, 1L, 2L);
            style.Ttl.Should().Equal(0.5f, -0.5f);
            style.DpShape.Should().Equal(1L, 1L, 3L);
            style.Dp.Should().Equal(1f, 2f, 3f);
        }
        finally { File.Delete(path); }
    }
}
