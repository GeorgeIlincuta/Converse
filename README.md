# Converse

A self-hosted, **fully local** voice API for **practising German**. Everything —
speech-to-text, the LLM reply, and text-to-speech — runs on your own machine
with no cloud dependencies.

Two use cases it's built for:

1. **Spoken German conversation** — record audio, get a spoken German reply, to
   train your German.
2. **Listen to German text** — send any German text and hear it spoken (e.g. a
   browser extension that reads selected text aloud).

Pipelines:

```
conversation:   audio in ─▶ Whisper (STT) ─▶ LM Studio (LLM) ─▶ Supertonic (TTS) ─▶ audio out
read-aloud:     text  in ─────────────────────────────────────▶ Supertonic (TTS) ─▶ audio out
```

German is the default language throughout, but the models are multilingual
(31 languages) and the language is configurable per request.

## Stack

- **.NET 10** minimal-API web service (`Converse.Api`)
- **Speech-to-text:** [Whisper.net](https://github.com/sandrohanea/whisper.net) running a local multilingual `ggml` model
- **LLM:** [LM Studio](https://lmstudio.ai/) via its OpenAI-compatible HTTP endpoint
- **Text-to-speech:** [Supertonic](https://huggingface.co/Supertone/supertonic-3) v1.7.3 (multilingual) — a 4-stage flow-matching ONNX pipeline run through [ONNX Runtime](https://onnxruntime.ai/)
- **GPU:** optional **DirectML** acceleration for the TTS models (any DX12 GPU), with automatic CPU fallback
- **Audio:** [NAudio](https://github.com/naudio/NAudio) for WAV/PCM conversion
- **Tests:** xUnit + FluentAssertions (`Converse.Api.Tests`)

## Project layout

```
Converse.Api/
  Audio/          WAV ⇄ PCM conversion (NAudio)
  Configuration/  Strongly-typed options (Whisper, Supertonic, Llm)
  Conversation/   Orchestrator, session store, domain models
  Endpoints/      Minimal-API route definitions (conversations, stt, tts)
  Llm/            ILlmService + LM Studio implementation
  Stt/            Whisper speech-to-text
  Tts/            Supertonic pipeline, text processor, voice loader,
                  helpers, and the startup warm-up service
  Program.cs      DI wiring, CORS, Kestrel config, /health probe
models/
  whisper/        Whisper ggml model       (gitignored)
  supertonic/     Supertonic ONNX models + config (gitignored)
  supertonic/voices/  Voice-style JSON files (gitignored)
```

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [LM Studio](https://lmstudio.ai/) running with a (German-capable) model loaded
  and its local server started — **only needed for the conversation flow**, not
  for `/tts`
- Model files placed on disk (see below)
- Optional: a DX12 GPU for DirectML acceleration (otherwise it runs on CPU)

### Model files

These are **not** committed to the repo — download and drop them in place.

**Whisper** (STT) — a multilingual ggml model at `models/whisper/ggml-small.bin`:

```bash
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin \
  -o models/whisper/ggml-small.bin
```
(Use a non-`.en` model for German. Larger = more accurate but slower; update
`Whisper:ModelPath` if you pick a different one.)

**Supertonic** (TTS) — the model set and voice styles from
[`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3):

`models/supertonic/`:

| File                      | Purpose                          |
|---------------------------|----------------------------------|
| `text_encoder.onnx`       | Text encoder                     |
| `duration_predictor.onnx` | Duration prediction              |
| `vector_estimator.onnx`   | Flow-matching vector estimator   |
| `vocoder.onnx`            | Vocoder                          |
| `tts.json`                | Model config (sample rate, etc.) |
| `unicode_indexer.json`    | Tokenizer vocabulary             |

`models/supertonic/voices/` — the voice styles (must match the model version):

```bash
for v in M1 M2 M3 M4 M5 F1 F2 F3 F4 F5; do
  curl -L "https://huggingface.co/Supertone/supertonic-3/resolve/main/voice_styles/$v.json" \
    -o "models/supertonic/voices/$v.json"
done
```

If any required model/voice is missing, the corresponding service reports
"not ready" via `/health` and the relevant endpoints return `503`.

## Configuration

Settings live in `Converse.Api/appsettings.json` (overridable per environment
and via environment variables, e.g. `Supertonic__CfmSteps=8`):

```jsonc
{
  "Whisper": {
    "ModelPath": "models/whisper/ggml-small.bin",
    "Language": "de"
  },
  "Supertonic": {
    "ModelsDirectory": "models/supertonic",
    "VoicesDirectory": "models/supertonic/voices",
    "DefaultVoice": "M1",     // M1–M5 (male), F1–F5 (female)
    "Language": "de",         // default <lang> tag for synthesis
    "CfmSteps": 8,            // flow-matching steps: higher = better, slower (8 ≈ 16 quality)
    "Speed": 1.05,            // speech speed (predicted duration / Speed)
    "UseGpu": true,           // DirectML GPU acceleration (CPU fallback if it fails)
    "GpuDeviceId": 0          // DirectML adapter index (discrete GPU may be 0 or 1)
  },
  "Llm": {
    "LmStudio": {
      "BaseUrl": "http://127.0.0.1:1234",
      "Model": "local-model",
      "ApiKey": ""
    }
  },
  "Kestrel": {
    "Endpoints": { "Http": { "Url": "http://127.0.0.1:5000" } }
  }
}
```

Notes:
- Only LM Studio is wired up as an LLM provider; others can go behind `ILlmService`.
- `LmStudio:Model` is passed through as-is; `"local-model"` works because LM
  Studio routes unknown ids to the currently loaded model.
- On a hybrid-GPU laptop, the discrete GPU may be `GpuDeviceId` 0 or 1 — pick the
  one that benchmarks fastest. Set `UseGpu: false` to run TTS entirely on CPU.

## Running

```bash
dotnet run --project Converse.Api
```

The API listens on `http://127.0.0.1:5000`. Audio uploads up to 50 MB are
accepted. **CORS** is enabled (any origin) so a browser extension can call it.

On startup a background **warm-up** runs one throwaway synthesis so the first
real request doesn't pay the model/DirectML initialization cost.

## Performance

TTS time is dominated by the flow-matching loop (`CfmSteps` × the vector
estimator) and scales with text length. Indicative steady-state numbers on an
RTX 5070 (DirectML, `CfmSteps: 8`):

| Input                | CPU    | GPU (DirectML) |
|----------------------|--------|----------------|
| Short sentence       | ~0.8 s | ~0.1 s         |
| ~500-char paragraph  | ~5 s   | ~1.7 s         |

## Tests

```bash
dotnet test
```

Integration tests that exercise the real ONNX pipeline are gated on the model
files being present (they skip if the models aren't installed).

## API

### `GET /health`
Readiness of each component.

```json
{ "whisper": true, "tts": true, "llm": true }
```

### `POST /tts`
Standalone text-to-speech. JSON body; returns a WAV file (`audio/wav`,
44.1 kHz mono). `voice` and `lang` are optional and default to config.

```json
{ "text": "Guten Tag!", "voice": "F1", "lang": "de" }
```

Returns `400` for an unknown voice or unsupported language, `503` if TTS isn't
ready.

### `POST /conversations`
Create a conversation session. JSON body (optional system prompt):

```json
{ "systemPrompt": "Du bist ein freundlicher Deutschlehrer. Antworte immer auf Deutsch." }
```

Returns `201 Created` with `{ "id": "<guid>" }`.

### `POST /conversations/{id}/turn`
The main loop. Send a `multipart/form-data` request with an `audio` file. The
server transcribes it, generates an LLM reply within the session's history, and
returns the spoken reply as a WAV file.

The transcript and reply text come back in URL-encoded response headers:

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

## Notes

- Conversation sessions are stored **in memory** (`InMemoryConversationStore`)
  and are lost on restart.
- All inference runs locally; no audio or text leaves your machine (aside from
  the local LM Studio HTTP call for conversations).
- Planned clients (built on top of this API): a Chrome extension that reads
  selected text aloud, and a desktop/mobile app for spoken conversation.
