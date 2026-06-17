using System.Diagnostics;
using Converse.Api.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Converse.Api.Tts;

public sealed class SupertonicPipeline : IDisposable
{
    private const string TextEncoderFile = "text_encoder.onnx";
    private const string DurationPredictorFile = "duration_predictor.onnx";
    private const string VectorEstimatorFile = "vector_estimator.onnx";
    private const string VocoderFile = "vocoder.onnx";
    private const string TtsJsonFile = "tts.json";
    private const string UnicodeIndexerFile = "unicode_indexer.json";

    private readonly ILogger<SupertonicPipeline> _logger;
    private readonly SupertonicOptions _opts;

    private readonly InferenceSession? _textEncoder;
    private readonly InferenceSession? _durationPredictor;
    private readonly InferenceSession? _vectorEstimator;
    private readonly InferenceSession? _vocoder;

    private readonly Dictionary<string, VoiceStyle> _voices = new();
    private readonly SupertonicTextProcessor? _processor;

    public TtsConfig? Config { get; }
    public UnicodeIndexer? Indexer { get; }
    public bool IsReady { get; }

    /// <summary>Convenience accessor; 0 if the pipeline did not load successfully.</summary>
    public int SampleRate => Config?.SampleRate ?? 0;

    public SupertonicPipeline(IOptions<SupertonicOptions> options, ILogger<SupertonicPipeline> logger)
    {
        _opts = options.Value;
        _logger = logger;

        var dir = Path.GetFullPath(_opts.ModelsDirectory);
        try
        {
            var ttsJsonPath = Path.Combine(dir, TtsJsonFile);
            var indexerPath = Path.Combine(dir, UnicodeIndexerFile);
            var textEncoderPath = Path.Combine(dir, TextEncoderFile);
            var durationPath = Path.Combine(dir, DurationPredictorFile);
            var vectorPath = Path.Combine(dir, VectorEstimatorFile);
            var vocoderPath = Path.Combine(dir, VocoderFile);

            foreach (var p in new[] { ttsJsonPath, indexerPath, textEncoderPath, durationPath, vectorPath, vocoderPath })
                if (!File.Exists(p))
                    throw new FileNotFoundException($"Required Supertonic file missing: {p}");

            Config = TtsConfig.Load(ttsJsonPath);
            Indexer = UnicodeIndexer.Load(indexerPath);

            _textEncoder = new InferenceSession(textEncoderPath);
            _durationPredictor = new InferenceSession(durationPath);
            _vectorEstimator = new InferenceSession(vectorPath);
            _vocoder = new InferenceSession(vocoderPath);

            var voicesDir = Path.GetFullPath(_opts.VoicesDirectory);
            foreach (var file in Directory.EnumerateFiles(voicesDir, "*.json"))
                _voices[Path.GetFileNameWithoutExtension(file)] = VoiceStyle.Load(file);
            if (!_voices.ContainsKey(_opts.DefaultVoice))
                throw new FileNotFoundException(
                    $"Default Supertonic voice '{_opts.DefaultVoice}' not found in {voicesDir}");
            _processor = new SupertonicTextProcessor(Indexer);

            _logger.LogInformation(
                "Supertonic loaded: 4 ONNX sessions, sample_rate={SampleRate}, latent_dim={LatentDim}, " +
                "chunk_compress_factor={Chunk}, hop_length={Hop}, cfm_steps={Steps}, " +
                "tts_version={Version}, split={Split}, vocab_size={Vocab}",
                Config.SampleRate, Config.LatentDim, Config.ChunkCompressFactor, Config.HopLength,
                _opts.CfmSteps, Config.TtsVersion, Config.Split, Indexer.VocabSize);

            LogSessionMetadata("text_encoder", _textEncoder);
            LogSessionMetadata("duration_predictor", _durationPredictor);
            LogSessionMetadata("vector_estimator", _vectorEstimator);
            LogSessionMetadata("vocoder", _vocoder);

            IsReady = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load Supertonic pipeline from '{Dir}'", dir);
            IsReady = false;
        }
    }

    public float[] Synthesize(string text, string? voice, string? lang, CancellationToken ct)
    {
        if (!IsReady || _processor is null || Config is null)
            throw new InvalidOperationException(
                "Supertonic pipeline is not ready; check models/voice directory configuration and startup logs.");

        var voiceName = string.IsNullOrWhiteSpace(voice) ? _opts.DefaultVoice : voice!;
        if (!_voices.TryGetValue(voiceName, out var style))
            throw new ArgumentException(
                $"Unknown voice '{voiceName}'. Available: {string.Join(", ", _voices.Keys.OrderBy(k => k))}.");

        var language = string.IsNullOrWhiteSpace(lang) ? _opts.Language : lang!;

        int maxLen = (language is "ko" or "ja") ? 120 : 300;
        var chunks = SupertonicHelpers.ChunkText(text, maxLen);

        const float silenceSeconds = 0.3f;
        var output = new List<float>();
        foreach (var chunk in chunks)
        {
            ct.ThrowIfCancellationRequested();
            var wav = InferChunk(chunk, style, language, ct);
            if (output.Count > 0)
                output.AddRange(new float[(int)(silenceSeconds * Config.SampleRate)]);
            output.AddRange(wav);
        }
        return output.ToArray();
    }

    private float[] InferChunk(string chunk, VoiceStyle voice, string lang, CancellationToken ct)
    {
        var (textIds, textMask) = _processor!.Encode(chunk, lang);
        int seq = textIds.Length;

        var textIdsTensor = new DenseTensor<long>(textIds, new[] { 1, seq });
        var textMaskTensor = new DenseTensor<float>(textMask, new[] { 1, 1, seq });
        var styleTtl = new DenseTensor<float>(voice.Ttl, voice.TtlShape.Select(x => (int)x).ToArray());
        var styleDp = new DenseTensor<float>(voice.Dp, voice.DpShape.Select(x => (int)x).ToArray());

        // 1) Duration predictor -> one total-duration value per utterance.
        var swDp = Stopwatch.StartNew();
        float[] duration;
        using (var dpOut = _durationPredictor!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
            NamedOnnxValue.CreateFromTensor("style_dp", styleDp),
            NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
        }))
        {
            duration = dpOut.First(o => o.Name == "duration").AsTensor<float>().ToArray();
        }
        for (int i = 0; i < duration.Length; i++)
            duration[i] /= _opts.Speed;
        swDp.Stop();

        // 2) Text encoder -> text_emb (copied out so we can dispose the run results).
        var swTe = Stopwatch.StartNew();
        DenseTensor<float> textEmb;
        using (var teOut = _textEncoder!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
            NamedOnnxValue.CreateFromTensor("style_ttl", styleTtl),
            NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
        }))
        {
            var t = teOut.First(o => o.Name == "text_emb").AsTensor<float>();
            textEmb = new DenseTensor<float>(t.ToArray(), t.Dimensions.ToArray());
        }
        swTe.Stop();

        // 3) Sample noisy latent + latent mask.
        float maxDuration = duration.Max();
        int latentLen = SupertonicHelpers.ComputeLatentLen(
            maxDuration, Config!.SampleRate, Config.BaseChunkSize, Config.ChunkCompressFactor);
        int latentDim = Config.LatentDim * Config.ChunkCompressFactor;

        long wavLength = (long)(maxDuration * Config.SampleRate);
        long latentLength = SupertonicHelpers.LatentLengthFor(
            wavLength, Config.BaseChunkSize, Config.ChunkCompressFactor);
        var latentMask = SupertonicHelpers.Mask(latentLength, latentLen);

        var xt = new float[latentDim * latentLen];
        var rng = new Random();
        for (int d = 0; d < latentDim; d++)
            for (int t = 0; t < latentLen; t++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                float sample = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                xt[d * latentLen + t] = sample * latentMask[t];
            }

        var latentMaskTensor = new DenseTensor<float>(latentMask, new[] { 1, 1, latentLen });
        int totalStep = _opts.CfmSteps;
        var totalStepTensor = new DenseTensor<float>(new[] { (float)totalStep }, new[] { 1 });

        // 4) Iterative denoising (no CFG).
        var swCfm = Stopwatch.StartNew();
        for (int step = 0; step < totalStep; step++)
        {
            ct.ThrowIfCancellationRequested();
            var noisy = new DenseTensor<float>(xt, new[] { 1, latentDim, latentLen });
            var currentStepTensor = new DenseTensor<float>(new[] { (float)step }, new[] { 1 });

            using var veOut = _vectorEstimator!.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("noisy_latent", noisy),
                NamedOnnxValue.CreateFromTensor("text_emb", textEmb),
                NamedOnnxValue.CreateFromTensor("style_ttl", styleTtl),
                NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
                NamedOnnxValue.CreateFromTensor("latent_mask", latentMaskTensor),
                NamedOnnxValue.CreateFromTensor("total_step", totalStepTensor),
                NamedOnnxValue.CreateFromTensor("current_step", currentStepTensor),
            });
            var denoised = veOut.First(o => o.Name == "denoised_latent").AsTensor<float>();
            var flat = denoised.ToArray();
            Array.Copy(flat, xt, xt.Length);
        }
        swCfm.Stop();

        // 5) Vocoder -> waveform.
        var swVoc = Stopwatch.StartNew();
        var latentTensor = new DenseTensor<float>(xt, new[] { 1, latentDim, latentLen });
        using var vocOut = _vocoder!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("latent", latentTensor),
        });
        var wav = vocOut.First(o => o.Name == "wav_tts").AsTensor<float>().ToArray();
        swVoc.Stop();

        _logger.LogInformation(
            "Supertonic timings (seq={Seq}, latentLen={LatentLen}, steps={Steps}): " +
            "dp={Dp}ms, textEnc={Te}ms, cfm={Cfm}ms ({PerStep:F0}ms/step), vocoder={Voc}ms, total={Total}ms",
            seq, latentLen, totalStep,
            swDp.ElapsedMilliseconds, swTe.ElapsedMilliseconds, swCfm.ElapsedMilliseconds,
            totalStep > 0 ? (double)swCfm.ElapsedMilliseconds / totalStep : 0,
            swVoc.ElapsedMilliseconds,
            swDp.ElapsedMilliseconds + swTe.ElapsedMilliseconds + swCfm.ElapsedMilliseconds + swVoc.ElapsedMilliseconds);

        return wav;
    }

    private void LogSessionMetadata(string name, InferenceSession session)
    {
        foreach (var kv in session.InputMetadata)
        {
            _logger.LogInformation(
                "Supertonic {Session} input '{Name}': type={ElementType}, shape={Shape}",
                name, kv.Key, kv.Value.ElementType, string.Join("x", kv.Value.Dimensions));
        }
        foreach (var kv in session.OutputMetadata)
        {
            _logger.LogInformation(
                "Supertonic {Session} output '{Name}': type={ElementType}, shape={Shape}",
                name, kv.Key, kv.Value.ElementType, string.Join("x", kv.Value.Dimensions));
        }
    }

    public void Dispose()
    {
        _textEncoder?.Dispose();
        _durationPredictor?.Dispose();
        _vectorEstimator?.Dispose();
        _vocoder?.Dispose();
    }
}
