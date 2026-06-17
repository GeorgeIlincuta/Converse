using FluentAssertions;
using Converse.Api.Configuration;
using Converse.Api.Tts;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace Converse.Api.Tests;

public class SupertonicPipelineSmokeTests
{
    // Walks up from the test output dir to find the repo's models/supertonic folder.
    private static string? FindModelsDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "supertonic", "tts.json");
            if (File.Exists(candidate)) return Path.Combine(dir.FullName, "models", "supertonic");
            dir = dir.Parent;
        }
        return null;
    }

    [Fact]
    public void Synthesize_produces_valid_german_audio()
    {
        var modelsDir = FindModelsDir();
        if (modelsDir is null || !File.Exists(Path.Combine(modelsDir, "voices", "M1.json")))
            return; // Models/voices not present in this environment — skip.

        var opts = Options.Create(new SupertonicOptions
        {
            ModelsDirectory = modelsDir,
            VoicesDirectory = Path.Combine(modelsDir, "voices"),
            DefaultVoice = "M1",
            Language = "de",
            CfmSteps = 16,
            Speed = 1.05f,
        });

        using var pipeline = new SupertonicPipeline(opts, NullLogger<SupertonicPipeline>.Instance);
        pipeline.IsReady.Should().BeTrue();

        var samples = pipeline.Synthesize("Hallo, wie geht es dir?", CancellationToken.None);

        samples.Should().NotBeEmpty();
        samples.Should().OnlyContain(s => !float.IsNaN(s) && !float.IsInfinity(s));
        samples.Should().OnlyContain(s => s >= -1f && s <= 1f);
        // At least ~0.3s of audio at 44.1 kHz for a short sentence.
        samples.Length.Should().BeGreaterThan((int)(0.3 * pipeline.SampleRate));
    }
}
