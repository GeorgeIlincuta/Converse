using System.Text.Json;

namespace Converse.Api.Tts;

public sealed class TtsConfig
{
    public required int SampleRate { get; init; }
    public required int LatentDim { get; init; }
    public required int ChunkCompressFactor { get; init; }
    public required int HopLength { get; init; }
    public required int BaseChunkSize { get; init; }
    public required float SigMin { get; init; }
    public required string TtsVersion { get; init; }
    public required string Split { get; init; }

    public static TtsConfig Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var ttl = root.GetProperty("ttl");
        var ae = root.GetProperty("ae");

        return new TtsConfig
        {
            TtsVersion = root.GetProperty("tts_version").GetString() ?? "",
            Split = root.GetProperty("split").GetString() ?? "",
            LatentDim = ttl.GetProperty("latent_dim").GetInt32(),
            ChunkCompressFactor = ttl.GetProperty("chunk_compress_factor").GetInt32(),
            SigMin = ttl.GetProperty("flow_matching").GetProperty("sig_min").GetSingle(),
            SampleRate = ae.GetProperty("sample_rate").GetInt32(),
            HopLength = ae.GetProperty("encoder").GetProperty("spec_processor").GetProperty("hop_length").GetInt32(),
            BaseChunkSize = ae.GetProperty("base_chunk_size").GetInt32(),
        };
    }
}
