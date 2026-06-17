namespace Converse.Api.Configuration;

public sealed class SupertonicOptions
{
    public const string SectionName = "Supertonic";

    public string ModelsDirectory { get; set; } = "models/supertonic";

    // Directory holding voice-style JSON files (e.g. M1.json).
    public string VoicesDirectory { get; set; } = "models/supertonic/voices";

    // Default voice-style file name (without extension).
    public string DefaultVoice { get; set; } = "M1";

    // Language code for synthesis (wrapped as <lang>…</lang>). German by default.
    public string Language { get; set; } = "de";

    // Flow-matching denoising step count (passed to the model as total_step).
    public int CfmSteps { get; set; } = 16;

    // Speech speed factor; predicted duration is divided by this. 1.05 matches the reference default.
    public float Speed { get; set; } = 1.05f;

    // Run ONNX inference on the GPU via the DirectML execution provider. Falls back to CPU
    // if DirectML initialization fails.
    public bool UseGpu { get; set; } = true;

    // DirectML adapter index. On hybrid-GPU laptops the discrete GPU may be index 0 or 1;
    // pick the one that benchmarks fastest.
    public int GpuDeviceId { get; set; } = 0;
}
