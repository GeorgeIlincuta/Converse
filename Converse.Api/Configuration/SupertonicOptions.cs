namespace Converse.Api.Configuration;

public sealed class SupertonicOptions
{
    public const string SectionName = "Supertonic";

    public string ModelsDirectory { get; set; } = "models/supertonic";

    // Flow-matching sampling step count. 16 is a reasonable default for CFM/F5-style vocoders;
    // raise for quality, lower for latency.
    public int CfmSteps { get; set; } = 16;

    // Classifier-free guidance scale. tts.json reveals the model was trained with prob_text_uncond.
    // 1.0 = no guidance; 1.5–3.0 typical.
    public float CfgScale { get; set; } = 1.5f;
}
