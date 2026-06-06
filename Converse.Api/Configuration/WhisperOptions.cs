namespace Converse.Api.Configuration;

public sealed class WhisperOptions
{
    public const string SectionName = "Whisper";
    public string ModelPath { get; set; } = "models/whisper/ggml-base.en.bin";
    public string Language { get; set; } = "en";
}
