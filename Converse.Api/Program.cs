using Converse.Api.Audio;
using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Converse.Api.Endpoints;
using Converse.Api.Llm;
using Converse.Api.Stt;
using Converse.Api.Tts;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

// Options
builder.Services.Configure<WhisperOptions>(builder.Configuration.GetSection(WhisperOptions.SectionName));
builder.Services.Configure<SupertonicOptions>(builder.Configuration.GetSection(SupertonicOptions.SectionName));
builder.Services.Configure<LlmOptions>(builder.Configuration.GetSection(LlmOptions.SectionName));

// Core services
builder.Services.AddSingleton<IConversationStore, InMemoryConversationStore>();
builder.Services.AddSingleton<IAudioConverter, NAudioConverter>();
builder.Services.AddSingleton<ISpeechToTextService, WhisperSpeechToTextService>();
builder.Services.AddSingleton<SupertonicPipeline>();
builder.Services.AddSingleton<ITextToSpeechService, SupertonicTextToSpeechService>();
builder.Services.AddHostedService<TtsWarmupService>();
builder.Services.AddScoped<ConversationOrchestrator>();

// LLM — v1 ships LM Studio only. Future providers go through ILlmService + a factory.
builder.Services.AddHttpClient<LmStudioLlmService>();
builder.Services.AddScoped<ILlmService>(sp => sp.GetRequiredService<LmStudioLlmService>());

// Health probe HttpClient (used by /health to ping LM Studio /v1/models)
builder.Services.AddHttpClient("llm-health", (sp, c) =>
{
    var opts = sp.GetRequiredService<IOptions<LlmOptions>>().Value;
    if (!string.IsNullOrWhiteSpace(opts.LmStudio.BaseUrl))
        c.BaseAddress = new Uri(opts.LmStudio.BaseUrl.TrimEnd('/') + "/");
    c.Timeout = TimeSpan.FromSeconds(5);
});

// Kestrel — raise body size limit for audio uploads (multipart form limit must match)
const long MaxAudioBytes = 50 * 1024 * 1024; // 50 MB
builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = MaxAudioBytes);
builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = MaxAudioBytes);

// CORS — allow browser/extension callers (local-only API; tighten origins later).
builder.Services.AddCors(options => options.AddDefaultPolicy(policy =>
    policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));

var app = builder.Build();

app.UseCors();

app.MapGet("/health", async (
    ISpeechToTextService stt,
    ITextToSpeechService tts,
    IHttpClientFactory httpFactory,
    CancellationToken ct) =>
{
    var llmReady = await ProbeLmStudioAsync(httpFactory, ct);
    return Results.Ok(new
    {
        whisper = stt.IsReady,
        tts = tts.IsReady,
        llm = llmReady,
    });
});

app.MapConversationEndpoints();
app.MapSttEndpoints();
app.MapTtsEndpoints();

app.Run();

static async Task<bool> ProbeLmStudioAsync(IHttpClientFactory factory, CancellationToken ct)
{
    var client = factory.CreateClient("llm-health");
    if (client.BaseAddress is null) return false;
    try
    {
        using var response = await client.GetAsync("v1/models", ct);
        return response.IsSuccessStatusCode;
    }
    catch
    {
        return false;
    }
}

public partial class Program { } // for WebApplicationFactory; harmless otherwise
