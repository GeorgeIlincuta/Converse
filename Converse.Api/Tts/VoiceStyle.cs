using System.Text.Json;

namespace Converse.Api.Tts;

// A single Supertonic voice style: the style_ttl and style_dp tensors loaded
// from a voice-style JSON file (batch size 1).
public sealed class VoiceStyle
{
    public float[] Ttl { get; }
    public long[] TtlShape { get; }
    public float[] Dp { get; }
    public long[] DpShape { get; }

    public VoiceStyle(float[] ttl, long[] ttlShape, float[] dp, long[] dpShape)
    {
        Ttl = ttl;
        TtlShape = ttlShape;
        Dp = dp;
        DpShape = dpShape;
    }

    public static VoiceStyle Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var (ttl, ttlShape) = ReadTensor(root.GetProperty("style_ttl"));
        var (dp, dpShape) = ReadTensor(root.GetProperty("style_dp"));
        return new VoiceStyle(ttl, ttlShape, dp, dpShape);
    }

    private static (float[] data, long[] dims) ReadTensor(JsonElement tensor)
    {
        var dims = new List<long>();
        foreach (var d in tensor.GetProperty("dims").EnumerateArray())
            dims.Add(d.GetInt64());

        var data = new List<float>();
        Flatten(tensor.GetProperty("data"), data);
        return (data.ToArray(), dims.ToArray());
    }

    private static void Flatten(JsonElement element, List<float> sink)
    {
        if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var child in element.EnumerateArray())
                Flatten(child, sink);
        }
        else
        {
            sink.Add(element.GetSingle());
        }
    }
}
