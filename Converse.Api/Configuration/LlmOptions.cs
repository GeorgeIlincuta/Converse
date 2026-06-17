namespace Converse.Api.Configuration;

public sealed class LlmOptions
{
    public const string SectionName = "Llm";
    public LmStudioOptions LmStudio { get; set; } = new();
}

public sealed class LmStudioOptions
{
    public string BaseUrl { get; set; } = "http://localhost:1234";
    public string Model { get; set; } = "local-model";
    public string ApiKey { get; set; } = "";
}
