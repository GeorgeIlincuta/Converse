# Converse

A self-hosted, fully local voice conversation API. Speak to it, and it speaks
back — speech-to-text, an LLM reply, and text-to-speech all run on your own
machine with no cloud dependencies.

The pipeline is:

```
audio in ──▶ Whisper (STT) ──▶ LM Studio (LLM) ──▶ Supertonic (TTS) ──▶ audio out
```

## Stack

- **.NET 10** minimal-API web service (`Converse.Api`)
- **Speech-to-text:** [Whisper.net](https://github.com/sandrohanea/whisper.net) running a local `ggml` model
- **LLM:** [LM Studio](https://lmstudio.ai/) via its OpenAI-compatible HTTP endpoint
- **Text-to-speech:** Supertonic ONNX models run through [ONNX Runtime](https://onnxruntime.ai/)
- **Audio:** [NAudio](https://github.com/naudio/NAudio) for WAV/PCM conversion
- **Tests:** xUnit (`Converse.Api.Tests`)

## Project layout

```
Converse.Api/
  Audio/          WAV ⇄ PCM conversion (NAudio)
  Configuration/  Strongly-typed options (Whisper, Supertonic, Llm)
  Conversation/   Orchestrator, session store, domain models
  Endpoints/      Minimal-API route definitions
  Llm/            ILlmService + LM Studio implementation
  Stt/            Whisper speech-to-text
  Tts/            Supertonic pipeline (4 ONNX sessions) + tokenizer
  Program.cs      DI wiring, Kestrel config, /health probe
models/
  whisper/        Whisper ggml model goes here
  supertonic/     Supertonic ONNX models + config go here
```

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [LM Studio](https://lmstudio.ai/) running with a model loaded and its local
  server started (default `http://localhost:1234`)
- Model files placed on disk (see below)

### Model files

These are **not** committed to the repo — download and drop them in place:

**Whisper** — a ggml model at `models/whisper/ggml-base.en.bin`
(any `ggml-*.bin` works; update `Whisper:ModelPath` if you use a different one).

**Supertonic** — place the full model set in `models/supertonic/`:

| File                      | Purpose                          |
|---------------------------|----------------------------------|
| `text_encoder.onnx`       | Text encoder                     |
| `duration_predictor.onnx` | Duration prediction              |
| `vector_estimator.onnx`   | Flow-matching vector estimator   |
| `vocoder.onnx`            | Vocoder                          |
| `tts.json`                | Model config (sample rate, etc.) |
| `unicode_indexer.json`    | Tokenizer vocabulary             |

If any model is missing the corresponding service reports "not ready" via
`/health` and the relevant endpoints return `503`.

## Configuration

Settings live in `Converse.Api/appsettings.json` (overridable per environment
and via environment variables):

```jsonc
{
  "Whisper": {
    "ModelPath": "models/whisper/ggml-base.en.bin",
    "Language": "en"
  },
  "Supertonic": {
    "ModelsDirectory": "models/supertonic",
    "CfmSteps": 16,      // flow-matching steps: higher = better quality, slower
    "CfgScale": 1.5      // classifier-free guidance scale (1.0 = off)
  },
  "Llm": {
    "LmStudio": {
      "BaseUrl": "http://localhost:1234",
      "Model": "local-model",
      "ApiKey": ""
    }
  },
  "Kestrel": {
    "Endpoints": { "Http": { "Url": "http://127.0.0.1:5000" } }
  }
}
```

> Only LM Studio is wired up as an LLM provider in this version. Additional
> providers can be added behind `ILlmService`.

## Running

```bash
dotnet run --project Converse.Api
```

The API listens on `http://127.0.0.1:5000` by default. Audio uploads up to
50 MB are accepted.

## Tests

```bash
dotnet test
```

## API

### `GET /health`
Readiness of each component.

```json
{ "whisper": true, "tts": true, "llm": true }
```

### `POST /conversations`
Create a conversation session. JSON body (optional system prompt):

```json
{ "systemPrompt": "You are a helpful assistant." }
```

Returns `201 Created` with `{ "id": "<guid>" }`.

### `POST /conversations/{id}/turn`
The main loop. Send a `multipart/form-data` request with an `audio` file. The
server transcribes it, generates an LLM reply within the session's history, and
returns the spoken reply as a WAV file (`audio/wav`).

The transcript and reply text are returned in URL-encoded response headers:

- `X-User-Transcript` — what you said
- `X-Assistant-Text` — what the assistant replied

Returns `404` if the session does not exist, `503` if STT/TTS aren't ready.

### `GET /conversations/{id}`
Fetch a session's system prompt, creation time, and full turn history.

### `DELETE /conversations/{id}`
Delete a session. Returns `204`.

### `POST /stt`
Standalone speech-to-text. `multipart/form-data` with an `audio` file; returns
`{ "text": "..." }`.

### `POST /tts`
Standalone text-to-speech. JSON body `{ "text": "..." }`; returns a WAV file.

## Notes

- Conversation sessions are stored **in memory** (`InMemoryConversationStore`)
  and are lost on restart.
- All inference runs locally; no audio or text leaves your machine (aside from
  the local LM Studio HTTP call).
