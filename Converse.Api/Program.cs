using Converse.Api.Audio;
using Converse.Api.Configuration;
using Converse.Api.Conversation;
using Converse.Api.Endpoints;
using Converse.Api.Llm;
using Converse.Api.Stt;
using Converse.Api.Tts;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

// Options
builder.Services.Configure<WhisperOptions>(builder.Configuration.GetSection(WhisperOptions.SectionName));
builder.Services.Configure<SupertonicOptions>(builder.Configuration.GetSection(SupertonicOptions.SectionName));
builder.Services.Configure<LlmOptions>(builder.Configuration.GetSection(LlmOptions.SectionName));

// Core services
builder.Services.AddSingleton<IConversationStore, InMemoryConversationStore>();
builder.Services.AddSingleton<IAudioConverter, NAudioConverter>();
builder.Services.AddSingleton<ISpeechToTextService, WhisperSpeechToTextService>();
builder.Services.AddSingleton<ITextToSpeechService, SupertonicTextToSpeechService>();
builder.Services.AddScoped<ConversationOrchestrator>();

// HTTP clients for LLM providers
builder.Services.AddHttpClient<GeminiLlmService>();
builder.Services.AddHttpClient<OpenAICompatibleLlmService>();

// Keyed registrations resolved at turn time
builder.Services.AddKeyedScoped<ILlmService>("gemini", (sp, _) => sp.GetRequiredService<GeminiLlmService>());
builder.Services.AddKeyedScoped<ILlmService>("openai-compatible", (sp, _) => sp.GetRequiredService<OpenAICompatibleLlmService>());

// Kestrel — raise body size limit for audio uploads (multipart form limit must match)
const long MaxAudioBytes = 50 * 1024 * 1024; // 50 MB
builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = MaxAudioBytes);
builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = MaxAudioBytes);

var app = builder.Build();

app.MapGet("/health", (ISpeechToTextService stt, ITextToSpeechService tts) =>
    Results.Ok(new { whisper = stt.IsReady, tts = tts.IsReady }));

app.MapConversationEndpoints();
app.MapSttEndpoints();
app.MapTtsEndpoints();

app.Run();

public partial class Program { } // for WebApplicationFactory; harmless otherwise
