namespace Converse.Api.Configuration;

public sealed class SupertonicOptions
{
    public const string SectionName = "Supertonic";
    public string ModelPath { get; set; } = "models/supertonic/model.onnx";
}
