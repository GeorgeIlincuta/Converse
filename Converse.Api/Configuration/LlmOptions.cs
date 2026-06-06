namespace Converse.Api.Configuration;

public sealed class LlmOptions
{
    public const string SectionName = "Llm";
    public string DefaultProvider { get; set; } = "openai-compatible";
    public GeminiOptions Gemini { get; set; } = new();
    public OpenAICompatibleOptions OpenAICompatible { get; set; } = new();
}

public sealed class GeminiOptions
{
    public string ApiKey { get; set; } = "";
    public string Model { get; set; } = "gemini-2.5-flash";
}

public sealed class OpenAICompatibleOptions
{
    public string BaseUrl { get; set; } = "http://localhost:1234";
    public string Model { get; set; } = "local-model";
    public string ApiKey { get; set; } = "";
}
