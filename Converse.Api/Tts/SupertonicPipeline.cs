using Converse.Api.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;

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

    private readonly VoiceStyle? _voice;
    private readonly SupertonicTextProcessor? _processor;

    public TtsConfig? Config { get; }
    public UnicodeIndexer? Indexer { get; }
    public bool IsReady { get; }

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

            var voicePath = Path.Combine(Path.GetFullPath(_opts.VoicesDirectory), _opts.DefaultVoice + ".json");
            if (!File.Exists(voicePath))
                throw new FileNotFoundException($"Required Supertonic voice file missing: {voicePath}");
            _voice = VoiceStyle.Load(voicePath);
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

    public float[] Synthesize(string text, CancellationToken ct)
    {
        if (!IsReady)
            throw new InvalidOperationException(
                "Supertonic pipeline is not ready; check models directory configuration and startup logs.");

        // TODO: implement the 4-stage pipeline.
        //
        //   1. tokenIds = Indexer.Encode(text)
        //   2. textHidden = textEncoder.Run({ <tokenIds>, <maybe style/lang ids> })
        //   3. durations = durationPredictor.Run({ <textHidden>, <maybe style/sentence ids> })
        //   4. expanded = length-regulate textHidden by durations
        //   5. CFM sampling loop (opts.CfmSteps iterations) over vectorEstimator with cfg_scale = opts.CfgScale
        //   6. waveform = vocoder.Run({ <latent> })
        //
        // The exact tensor input names / shapes / dtypes are logged at startup.
        // Inspect the startup output before wiring this method — guesses will produce garbage audio
        // or runtime crashes (ORT validates shape strictly).
        throw new NotImplementedException(
            "Supertonic synthesis is scaffolded but not yet wired. " +
            "Run the app once and read the 'Supertonic loaded' / 'session' tensor metadata in the logs, " +
            "then implement the 4-stage pipeline using those exact input names.");
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
