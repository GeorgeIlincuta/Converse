# Supertonic TTS Synthesis (German-first) — Design

**Date:** 2026-06-17
**Status:** Approved design, pending spec review
**Component:** `Converse.Api/Tts`

## Context & purpose

Converse is a local voice API for **practicing/learning German**. Its first two
use cases — spoken German conversation (STT → LLM → TTS) and "listen to selected
German text" — both depend on working German text-to-speech. The TTS models
(Supertonic v1.7.3, opensource-multilingual) are already downloaded and load at
startup, but `SupertonicPipeline.Synthesize()` is an unimplemented scaffold that
throws `NotImplementedException`. This spec covers implementing that synthesis.

Out of scope (future phases): German STT (the shipped Whisper model is
English-only `ggml-base.en.bin`), the desktop/mobile app, and the Chrome
extension. This spec is scoped strictly to German-capable TTS synthesis.

## Reference

The implementation is a faithful port of the official Supertone C# reference
(`supertone-inc/supertonic`, `csharp/Helper.cs` + `csharp/ExampleONNX.cs`).
Models and voice styles come from the matching Hugging Face repo
`Supertone/supertonic-3` (`tts_version v1.7.3`, `split opensource-multilingual`).

**Implementation principle:** the algorithm details below were gathered from
documentation/summaries, which are lossy. During implementation the actual
reference source MUST be read verbatim — exact mask construction, the normalizer
`scale` (0.25), the CFM time schedule, and the NFKD handling — and cross-checked
against the ONNX models' logged input/output tensor metadata. No guessing on
tensor shapes or names.

## Architecture & data flow

No endpoint or interface changes. `/tts` already calls
`ITextToSpeechService.SynthesizeAsync(text)` and wraps the result with the
existing `IAudioConverter.PcmToWav(samples, sampleRate)`.

```
/tts (text) → SupertonicTextToSpeechService.SynthesizeAsync
            → SupertonicPipeline.Synthesize(text) : float[]
            → IAudioConverter.PcmToWav → WAV (44.1 kHz mono 16-bit)
```

`Synthesize` implements the five stages:

1. **Tokenize.** Normalize text (`SupertonicTextProcessor`) and map to ids via
   the existing `UnicodeIndexer`. Build `text_ids` `[1, seq]` and
   `text_mask` `[1, 1, seq]`.
2. **Duration.** `duration_predictor(text_ids, style_dp, text_mask)` → `duration`
   `[1, seq]`. Apply `Speed` (`duration[i] /= Speed`). Derive `latent_len` from
   total predicted duration, sample rate, and `chunk` size (exact formula taken
   from reference source).
3. **Text encode.** `text_encoder(text_ids, style_ttl, text_mask)` → `text_emb`.
4. **CFM sampling loop.** Seed `noisy_latent` `[1, latent_dim, latent_len]` from
   N(0,1) (seedable RNG for test determinism). Loop `CfmSteps` times calling
   `vector_estimator(noisy_latent, text_emb, style_ttl, text_mask, latent_mask,
   total_step, current_step)`, replacing the latent with `denoised_latent` each
   step.
5. **Vocode.** `vocoder(latent)` → `wav_tts`; return as `float[]`.

### ONNX tensor contract (verify against logged metadata at implementation time)

| Model | Inputs | Output |
|-------|--------|--------|
| `duration_predictor` | `text_ids`, `style_dp`, `text_mask` | `duration` |
| `text_encoder` | `text_ids`, `style_ttl`, `text_mask` | `text_emb` |
| `vector_estimator` | `noisy_latent`, `text_emb`, `style_ttl`, `text_mask`, `latent_mask`, `total_step`, `current_step` | `denoised_latent` |
| `vocoder` | `latent` | `wav_tts` |

## Components

### New

- **`VoiceStyle`** — loads a voice JSON (`{ style_ttl: {dims, data}, style_dp:
  {dims, data} }`) into `DenseTensor<float>` for `style_ttl` and `style_dp`.
  Loaded once at startup for the configured default voice.
- **`SupertonicTextProcessor`** — text normalization matching the reference
  `UnicodeProcessor`: Unicode NFKD, emoji removal, symbol replacement,
  punctuation spacing, and `<lang>…</lang>` wrapping using the configured
  language. Output feeds `UnicodeIndexer.Encode`. Kept separate so
  `SupertonicPipeline` stays focused on inference.

### Changed

- **`SupertonicPipeline`** — implement `Synthesize`; load the default
  `VoiceStyle` at construction; expose readiness reflecting voice availability.
- **`SupertonicOptions`** — add `VoicesDirectory` (default
  `models/supertonic/voices`), `DefaultVoice` (default `"M1"`), `Language`
  (default `"de"`), `Speed` (default `1.05`). Keep `CfmSteps` (→ `total_step`).
  Re-verify whether `CfgScale` is used by this export; remove it if the
  vector_estimator graph has no guidance input.

### Assets

- Download `Supertone/supertonic-3` `voice_styles/*.json` (M1–M5, F1–F5) into
  `models/supertonic/voices/`. Gitignored like the ONNX models.

## German correctness

German is the primary language (`lang="de"`, officially supported). The NFKD
normalization decomposes `ä → a + ̈`, `ö`, `ü`, and affects `ß`. The correct
form (decomposed vs. precomposed) is whatever the `unicode_indexer.json` table
encodes; this MUST be confirmed against the table and reference, and covered by
explicit umlaut/`ß` test cases. Garbled German output is the primary risk to
guard against.

## Error handling

A missing or unparseable default voice file, or missing models, leaves the
pipeline `IsReady = false` with a clear startup log (same pattern as the
existing model checks). `/tts` then returns `503`, never an unhandled throw.
A request for an unknown voice or an invalid language returns `400`.

## Voice / language selection

All preset voices (M1–M5, F1–F5) load at startup; `DefaultVoice` (M1) and
`Language` (de) are the config defaults. `/tts` accepts **optional per-request
`voice` and `lang`** fields so the Chrome extension can choose them per call;
when omitted, the defaults apply. An unknown voice or unsupported language is a
`400`. The conversation (`/turn`) path always uses the defaults (it calls the
parameterless overload), so it is unaffected.

## CORS (browser/extension access)

The API enables CORS so a Chrome extension can call `/tts` from the browser
(content-script requests are subject to cross-origin rules). A permissive
default policy (any origin) is acceptable for this local-only API and can be
tightened to the extension's `chrome-extension://<id>` origin later.

## Testing

- **`SupertonicTextProcessor`** — deterministic normalization/tokenization
  cases, including German umlauts and `ß`.
- **`VoiceStyle` loader** — parses `dims`/`data` into correctly shaped tensors;
  rejects malformed JSON.
- **Length regulation** — pure unit test of the duration→`latent_len` math.
- **Integration smoke test** (gated on model + voice files present) — runs the
  real pipeline on a short German phrase; asserts output is non-empty, all
  samples finite (no NaN/Inf) and within `[-1, 1]`, and length is plausible for
  the input. Subjective correctness verified by listening to a generated WAV.
- **Voice override** — a non-default voice synthesizes; an unknown voice raises
  `ArgumentException` (mapped to `400`).
- **CORS** — a request carrying an `Origin` header comes back with an
  `Access-Control-Allow-Origin` header.

## Success criteria

`POST /tts` with German text returns a valid 44.1 kHz mono WAV containing
intelligible German speech in the M1 voice (with umlauts and `ß` pronounced
correctly), honours optional `voice`/`lang` overrides, is reachable
cross-origin from a browser extension, and `/health` continues to report
`tts: true`.
