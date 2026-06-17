using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class TtsConfigTests
{
    private const string MinimalTtsJson = """
    {
      "tts_version": "v1.7.3",
      "split": "opensource-multilingual",
      "ttl": { "latent_dim": 24, "chunk_compress_factor": 6, "flow_matching": { "sig_min": 1e-08 } },
      "ae": {
        "sample_rate": 44100,
        "base_chunk_size": 512,
        "encoder": { "spec_processor": { "hop_length": 512 } }
      }
    }
    """;

    [Fact]
    public void Load_reads_base_chunk_size()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, MinimalTtsJson);
        try
        {
            var cfg = TtsConfig.Load(path);
            cfg.BaseChunkSize.Should().Be(512);
            cfg.SampleRate.Should().Be(44100);
            cfg.ChunkCompressFactor.Should().Be(6);
            cfg.LatentDim.Should().Be(24);
        }
        finally { File.Delete(path); }
    }
}
