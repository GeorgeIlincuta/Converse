# Supertonic TTS Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SupertonicPipeline.Synthesize` so `POST /tts` returns intelligible German speech, by faithfully porting the official Supertonic v1.7.3 ONNX inference pipeline.

**Architecture:** Port the official `supertone-inc/supertonic` C# reference (`csharp/Helper.cs`). Text → normalize/tokenize → `duration_predictor` → `text_encoder` → CFM denoising loop over `vector_estimator` → `vocoder` → `float[]` samples, wrapped to WAV by the existing `IAudioConverter.PcmToWav`. Pure logic (text processing, masks, voice-style loading, chunking) is split into small `public` classes that are unit-tested; ONNX orchestration stays in `SupertonicPipeline` and is covered by a gated integration test.

**Tech Stack:** .NET 10, Microsoft.ML.OnnxRuntime (`DenseTensor<T>`), xUnit + FluentAssertions. Models/voices from Hugging Face `Supertone/supertonic-3` (`tts_version v1.7.3`, `opensource-multilingual`).

**Reference correctness notes (do not deviate without checking the model):**
- Tokenization iterates **UTF-16 chars** (`(int)c`) of the *normalized* string. Unmapped chars become token **`0`** (NOT dropped). The text mask length equals the processed string length.
- Text is NFKD-normalized, so `ä`/`ö`/`ü` decompose to base letter + combining diaeresis (`̈`); the indexer is expected to contain entries for the components.
- `duration_predictor` outputs **one total-duration value (seconds) per utterance**, not per token.
- There is **no classifier-free guidance**: `vector_estimator` is called once per step. `total_step` and `current_step` are passed as `float` tensors of shape `[batch]`.
- `latentLen = ceil(maxDuration·sampleRate / (base_chunk_size·chunk_compress_factor))`; `latentDim = latent_dim·chunk_compress_factor`.
- Exact ONNX tensor names: `text_ids`, `style_dp`, `text_mask` → `duration`; `text_ids`, `style_ttl`, `text_mask` → `text_emb`; `noisy_latent`, `text_emb`, `style_ttl`, `text_mask`, `latent_mask`, `total_step`, `current_step` → `denoised_latent`; `latent` → `wav_tts`.

The verbatim reference is saved at `.refsrc/Helper.cs` / `.refsrc/ExampleONNX.cs` for consultation during implementation. Delete `.refsrc/` before the final commit (Task 11).

---

## Task 0: Pre-flight — download voices and verify tensor metadata

**Files:**
- Create: `models/supertonic/voices/*.json` (downloaded assets, gitignored)

- [ ] **Step 1: Download the 10 preset voice styles for v1.7.3**

Run (Git Bash):
```bash
cd "C:/LOCAL FILES/Claude Code/DotNet/Converse" && mkdir -p models/supertonic/voices && \
for v in M1 M2 M3 M4 M5 F1 F2 F3 F4 F5; do \
  curl -sL "https://huggingface.co/Supertone/supertonic-3/resolve/main/voice_styles/$v.json" \
    -o "models/supertonic/voices/$v.json"; \
  echo "$v: $(wc -c < models/supertonic/voices/$v.json) bytes"; \
done
```
Expected: each file ~hundreds of KB (non-zero). `models/supertonic/voices/` is already covered by the `/models/**` .gitignore rule — do not commit these.

- [ ] **Step 2: Confirm the running model's tensor names/shapes match the reference**

Stop any running instance, then run the API and trigger pipeline load:
```bash
cd "C:/LOCAL FILES/Claude Code/DotNet/Converse" && ASPNETCORE_ENVIRONMENT=Development dotnet run --project Converse.Api > .refsrc/boot.log 2>&1 &
sleep 12 && curl -s http://127.0.0.1:5000/health > /dev/null; sleep 2
grep -i "Supertonic .* input\|Supertonic .* output" .refsrc/boot.log
```
Expected: the logged input/output names include exactly `text_ids`, `style_dp`, `style_ttl`, `text_mask`, `duration`, `text_emb`, `noisy_latent`, `latent_mask`, `total_step`, `current_step`, `denoised_latent`, `latent`, `wav_tts`. If any name differs, STOP and reconcile the plan's tensor names with the log before proceeding. Stop the app afterward.

- [ ] **Step 3: Commit nothing** (downloads are gitignored; this task produces no tracked changes).

---

## Task 1: Add `base_chunk_size` to `TtsConfig`

**Files:**
- Modify: `Converse.Api/Tts/TtsConfig.cs`
- Test: `Converse.Api.Tests/TtsConfigTests.cs`

- [ ] **Step 1: Write the failing test**

Create `Converse.Api.Tests/TtsConfigTests.cs`:
```csharp
using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class TtsConfigTests
{
    private const string MinimalTtsJson = """
    {
      "tts_version": "v1.7.3",
      "split": "opensource-multilingual",
      "ttl": { "latent_dim": 24, "chunk_compress_factor": 6, "flow_matching": { "sig_min": 1e-08 } },
      "ae": {
        "sample_rate": 44100,
        "base_chunk_size": 512,
        "encoder": { "spec_processor": { "hop_length": 512 } }
      }
    }
    """;

    [Fact]
    public void Load_reads_base_chunk_size()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, MinimalTtsJson);
        try
        {
            var cfg = TtsConfig.Load(path);
            cfg.BaseChunkSize.Should().Be(512);
            cfg.SampleRate.Should().Be(44100);
            cfg.ChunkCompressFactor.Should().Be(6);
            cfg.LatentDim.Should().Be(24);
        }
        finally { File.Delete(path); }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter FullyQualifiedName~TtsConfigTests`
Expected: FAIL — `TtsConfig` has no `BaseChunkSize`.

- [ ] **Step 3: Add the property and parse it**

In `Converse.Api/Tts/TtsConfig.cs`, add the property next to the others:
```csharp
    public required int BaseChunkSize { get; init; }
```
And in `Load`, inside the returned `new TtsConfig { ... }`, add:
```csharp
            BaseChunkSize = ae.GetProperty("base_chunk_size").GetInt32(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter FullyQualifiedName~TtsConfigTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Tts/TtsConfig.cs Converse.Api.Tests/TtsConfigTests.cs
git commit -m "Add base_chunk_size to TtsConfig"
```

---

## Task 2: `UnicodeIndexer.MapChar` (replace `Encode`)

**Files:**
- Modify: `Converse.Api/Tts/UnicodeIndexer.cs`
- Test: `Converse.Api.Tests/UnicodeIndexerTests.cs`

- [ ] **Step 1: Write the failing test**

Create `Converse.Api.Tests/UnicodeIndexerTests.cs`:
```csharp
using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class UnicodeIndexerTests
{
    [Fact]
    public void MapChar_returns_table_value_for_in_range_codepoints()
    {
        var indexer = new UnicodeIndexer(new[] { 5, 7, 9 });
        indexer.MapChar(0).Should().Be(5L);
        indexer.MapChar(2).Should().Be(9L);
    }

    [Fact]
    public void MapChar_returns_zero_for_out_of_range_codepoints()
    {
        var indexer = new UnicodeIndexer(new[] { 5, 7, 9 });
        indexer.MapChar(3).Should().Be(0L);
        indexer.MapChar(-1).Should().Be(0L);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter FullyQualifiedName~UnicodeIndexerTests`
Expected: FAIL — `MapChar` not defined.

- [ ] **Step 3: Add `MapChar`, remove the unused `Encode`**

In `Converse.Api/Tts/UnicodeIndexer.cs`, delete the `Encode` method (and its comment block, lines ~30-48). Add:
```csharp
    // Maps a single UTF-16 code unit (cast to int) to its token id, matching the
    // reference: in-range codepoints return the table value, everything else returns 0.
    public long MapChar(int charValue)
        => charValue >= 0 && charValue < _table.Length ? _table[charValue] : 0L;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter FullyQualifiedName~UnicodeIndexerTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Tts/UnicodeIndexer.cs Converse.Api.Tests/UnicodeIndexerTests.cs
git commit -m "Add UnicodeIndexer.MapChar; drop unused Encode"
```

---

## Task 3: `SupertonicTextProcessor` (normalize + tokenize)

**Files:**
- Create: `Converse.Api/Tts/SupertonicTextProcessor.cs`
- Test: `Converse.Api.Tests/SupertonicTextProcessorTests.cs`

- [ ] **Step 1: Write the failing test**

Create `Converse.Api.Tests/SupertonicTextProcessorTests.cs`:
```csharp
using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class SupertonicTextProcessorTests
{
    // Identity-ish indexer big enough to cover ASCII + Latin-1 + combining marks.
    private static SupertonicTextProcessor MakeProcessor()
    {
        var table = new int[0x400];
        for (int i = 0; i < table.Length; i++) table[i] = i;
        return new SupertonicTextProcessor(new UnicodeIndexer(table));
    }

    [Fact]
    public void Preprocess_wraps_with_language_tags_and_appends_period()
    {
        MakeProcessor().Preprocess("Hallo", "de").Should().Be("<de>Hallo.</de>");
    }

    [Fact]
    public void Preprocess_does_not_append_period_when_already_punctuated()
    {
        MakeProcessor().Preprocess("Wie geht es?", "de").Should().Be("<de>Wie geht es?</de>");
    }

    [Fact]
    public void Preprocess_decomposes_umlauts_via_nfkd()
    {
        var result = MakeProcessor().Preprocess("schön", "de");
        result.Should().NotContain("ö");   // precomposed o-umlaut removed by NFKD
        result.Should().Contain("̈");      // combining diaeresis is present
    }

    [Fact]
    public void Preprocess_throws_on_invalid_language()
    {
        var act = () => MakeProcessor().Preprocess("hi", "xx");
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Encode_produces_ids_and_all_ones_mask_of_processed_length()
    {
        var p = MakeProcessor();
        var processed = p.Preprocess("Hallo", "de");
        var (ids, mask) = p.Encode("Hallo", "de");

        ids.Should().HaveCount(processed.Length);
        mask.Should().HaveCount(processed.Length);
        mask.Should().OnlyContain(x => x == 1f);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter FullyQualifiedName~SupertonicTextProcessorTests`
Expected: FAIL — type does not exist.

- [ ] **Step 3: Implement `SupertonicTextProcessor`**

Create `Converse.Api/Tts/SupertonicTextProcessor.cs` (ported from the reference `UnicodeProcessor`):
```csharp
using System.Text;
using System.Text.RegularExpressions;

namespace Converse.Api.Tts;

// Normalizes and tokenizes input text for Supertonic, ported from the official
// reference UnicodeProcessor. Tokenization iterates UTF-16 code units; unmapped
// characters become token id 0, and the mask length equals the processed length.
public sealed class SupertonicTextProcessor
{
    private static readonly string[] AvailableLanguages =
    {
        "en","ko","ja","ar","bg","cs","da","de","el","es","et","fi","fr","hi","hr",
        "hu","id","it","lt","lv","nl","pl","pt","ro","ru","sk","sl","sv","tr","uk","vi","na"
    };

    private static readonly Dictionary<string, string> Replacements = new()
    {
        {"–", "-"}, {"‑", "-"}, {"—", "-"}, {"_", " "},
        {"“", "\""}, {"”", "\""}, {"‘", "'"}, {"’", "'"},
        {"´", "'"}, {"`", "'"}, {"[", " "}, {"]", " "}, {"|", " "}, {"/", " "},
        {"#", " "}, {"→", " "}, {"←", " "},
    };

    private static readonly Dictionary<string, string> ExpressionReplacements = new()
    {
        {"@", " at "}, {"e.g.,", "for example, "}, {"i.e.,", "that is, "},
    };

    private readonly UnicodeIndexer _indexer;

    public SupertonicTextProcessor(UnicodeIndexer indexer) => _indexer = indexer;

    public string Preprocess(string text, string lang)
    {
        text = text.Normalize(NormalizationForm.FormKD);
        text = RemoveEmojis(text);

        foreach (var kvp in Replacements)
            text = text.Replace(kvp.Key, kvp.Value);

        text = Regex.Replace(text, @"[♥☆♡©\\]", "");

        foreach (var kvp in ExpressionReplacements)
            text = text.Replace(kvp.Key, kvp.Value);

        text = Regex.Replace(text, @" ,", ",");
        text = Regex.Replace(text, @" \.", ".");
        text = Regex.Replace(text, @" !", "!");
        text = Regex.Replace(text, @" \?", "?");
        text = Regex.Replace(text, @" ;", ";");
        text = Regex.Replace(text, @" :", ":");
        text = Regex.Replace(text, @" '", "'");

        while (text.Contains("\"\"")) text = text.Replace("\"\"", "\"");
        while (text.Contains("''")) text = text.Replace("''", "'");
        while (text.Contains("``")) text = text.Replace("``", "`");

        text = Regex.Replace(text, @"\s+", " ").Trim();

        // Append a period unless the text already ends with sentence punctuation,
        // a quote, or a closing bracket. Using IndexOf over a normal string avoids
        // the quote-escaping pitfalls of a verbatim regex character class.
        const string EndingChars = ".!?;:,'\"“”‘’)]}…。」』】〉》›»";
        if (text.Length == 0 || EndingChars.IndexOf(text[^1]) < 0)
            text += ".";

        if (!AvailableLanguages.Contains(lang))
            throw new ArgumentException(
                $"Invalid language: {lang}. Available: {string.Join(", ", AvailableLanguages)}");

        return $"<{lang}>{text}</{lang}>";
    }

    public (long[] textIds, float[] textMask) Encode(string text, string lang)
    {
        var processed = Preprocess(text, lang);
        var ids = new long[processed.Length];
        var mask = new float[processed.Length];
        for (int i = 0; i < processed.Length; i++)
        {
            ids[i] = _indexer.MapChar(processed[i]);
            mask[i] = 1f;
        }
        return (ids, mask);
    }

    private static string RemoveEmojis(string text)
    {
        var result = new StringBuilder();
        for (int i = 0; i < text.Length; i++)
        {
            int codePoint;
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                codePoint = char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }
            else
            {
                codePoint = text[i];
            }

            bool isEmoji =
                (codePoint >= 0x1F600 && codePoint <= 0x1F64F) ||
                (codePoint >= 0x1F300 && codePoint <= 0x1F5FF) ||
                (codePoint >= 0x1F680 && codePoint <= 0x1F6FF) ||
                (codePoint >= 0x1F700 && codePoint <= 0x1F77F) ||
                (codePoint >= 0x1F780 && codePoint <= 0x1F7FF) ||
                (codePoint >= 0x1F800 && codePoint <= 0x1F8FF) ||
                (codePoint >= 0x1F900 && codePoint <= 0x1F9FF) ||
                (codePoint >= 0x1FA00 && codePoint <= 0x1FA6F) ||
                (codePoint >= 0x1FA70 && codePoint <= 0x1FAFF) ||
                (codePoint >= 0x2600 && codePoint <= 0x26FF) ||
                (codePoint >= 0x2700 && codePoint <= 0x27BF) ||
                (codePoint >= 0x1F1E6 && codePoint <= 0x1F1FF);

            if (!isEmoji)
                result.Append(codePoint > 0xFFFF ? char.ConvertFromUtf32(codePoint) : ((char)codePoint).ToString());
        }
        return result.ToString();
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter FullyQualifiedName~SupertonicTextProcessorTests`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Tts/SupertonicTextProcessor.cs Converse.Api.Tests/SupertonicTextProcessorTests.cs
git commit -m "Add SupertonicTextProcessor (NFKD normalize + tokenize)"
```

---

## Task 4: `VoiceStyle` loader

**Files:**
- Create: `Converse.Api/Tts/VoiceStyle.cs`
- Test: `Converse.Api.Tests/VoiceStyleTests.cs`

- [ ] **Step 1: Write the failing test**

Create `Converse.Api.Tests/VoiceStyleTests.cs`:
```csharp
using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class VoiceStyleTests
{
    private const string Json = """
    {
      "style_ttl": { "dims": [1, 1, 2], "data": [[[0.5, -0.5]]], "type": "float32" },
      "style_dp":  { "dims": [1, 1, 3], "data": [[[1.0, 2.0, 3.0]]], "type": "float32" },
      "metadata":  { "source_file": "x" }
    }
    """;

    [Fact]
    public void Load_parses_shapes_and_flattened_data()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, Json);
        try
        {
            var style = VoiceStyle.Load(path);
            style.TtlShape.Should().Equal(1L, 1L, 2L);
            style.Ttl.Should().Equal(0.5f, -0.5f);
            style.DpShape.Should().Equal(1L, 1L, 3L);
            style.Dp.Should().Equal(1f, 2f, 3f);
        }
        finally { File.Delete(path); }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter FullyQualifiedName~VoiceStyleTests`
Expected: FAIL — type does not exist.

- [ ] **Step 3: Implement `VoiceStyle`**

Create `Converse.Api/Tts/VoiceStyle.cs` (single-voice port of the reference `LoadVoiceStyle`):
```csharp
using System.Text.Json;

namespace Converse.Api.Tts;

// A single Supertonic voice style: the style_ttl and style_dp tensors loaded
// from a voice-style JSON file (batch size 1).
public sealed class VoiceStyle
{
    public float[] Ttl { get; }
    public long[] TtlShape { get; }
    public float[] Dp { get; }
    public long[] DpShape { get; }

    public VoiceStyle(float[] ttl, long[] ttlShape, float[] dp, long[] dpShape)
    {
        Ttl = ttl;
        TtlShape = ttlShape;
        Dp = dp;
        DpShape = dpShape;
    }

    public static VoiceStyle Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var (ttl, ttlShape) = ReadTensor(root.GetProperty("style_ttl"));
        var (dp, dpShape) = ReadTensor(root.GetProperty("style_dp"));
        return new VoiceStyle(ttl, ttlShape, dp, dpShape);
    }

    private static (float[] data, long[] dims) ReadTensor(JsonElement tensor)
    {
        var dims = new List<long>();
        foreach (var d in tensor.GetProperty("dims").EnumerateArray())
            dims.Add(d.GetInt64());

        var data = new List<float>();
        Flatten(tensor.GetProperty("data"), data);
        return (data.ToArray(), dims.ToArray());
    }

    private static void Flatten(JsonElement element, List<float> sink)
    {
        if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var child in element.EnumerateArray())
                Flatten(child, sink);
        }
        else
        {
            sink.Add(element.GetSingle());
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter FullyQualifiedName~VoiceStyleTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Tts/VoiceStyle.cs Converse.Api.Tests/VoiceStyleTests.cs
git commit -m "Add VoiceStyle loader"
```

---

## Task 5: `SupertonicHelpers` (masks, latent length, chunking, tensor builders)

**Files:**
- Create: `Converse.Api/Tts/SupertonicHelpers.cs`
- Test: `Converse.Api.Tests/SupertonicHelpersTests.cs`

- [ ] **Step 1: Write the failing test**

Create `Converse.Api.Tests/SupertonicHelpersTests.cs`:
```csharp
using FluentAssertions;
using Converse.Api.Tts;

namespace Converse.Api.Tests;

public class SupertonicHelpersTests
{
    [Fact]
    public void ComputeLatentLen_ceils_to_chunk_size()
    {
        // chunkSize = 512 * 6 = 3072; wavLenMax = 1.0 * 44100 = 44100; ceil(44100/3072) = 15
        SupertonicHelpers.ComputeLatentLen(1.0f, 44100, 512, 6).Should().Be(15);
    }

    [Fact]
    public void LatentLengthFor_ceils_wav_length()
    {
        // latentSize = 3072; ceil(44100/3072) = 15
        SupertonicHelpers.LatentLengthFor(44100, 512, 6).Should().Be(15);
    }

    [Fact]
    public void Mask_is_ones_then_zeros()
    {
        var mask = SupertonicHelpers.Mask(length: 3, total: 5);
        mask.Should().Equal(1f, 1f, 1f, 0f, 0f);
    }

    [Fact]
    public void ChunkText_returns_single_chunk_for_short_text()
    {
        var chunks = SupertonicHelpers.ChunkText("Hallo, wie geht es dir?", 300);
        chunks.Should().ContainSingle();
    }

    [Fact]
    public void ChunkText_splits_long_text_into_multiple_chunks()
    {
        var sentence = "Dies ist ein deutscher Satz. ";
        var text = string.Concat(Enumerable.Repeat(sentence, 40)); // > 300 chars
        var chunks = SupertonicHelpers.ChunkText(text, 100);
        chunks.Count.Should().BeGreaterThan(1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter FullyQualifiedName~SupertonicHelpersTests`
Expected: FAIL — type does not exist.

- [ ] **Step 3: Implement `SupertonicHelpers`**

Create `Converse.Api/Tts/SupertonicHelpers.cs`:
```csharp
using System.Text.RegularExpressions;

namespace Converse.Api.Tts;

// Pure helpers ported from the reference Helper class: latent-length math,
// boolean length masks, and sentence-aware text chunking.
public static class SupertonicHelpers
{
    public static int ComputeLatentLen(float maxDurationSeconds, int sampleRate, int baseChunkSize, int chunkCompressFactor)
    {
        int chunkSize = baseChunkSize * chunkCompressFactor;
        float wavLenMax = maxDurationSeconds * sampleRate;
        return (int)((wavLenMax + chunkSize - 1) / chunkSize);
    }

    public static long LatentLengthFor(long wavLength, int baseChunkSize, int chunkCompressFactor)
    {
        long latentSize = baseChunkSize * chunkCompressFactor;
        return (wavLength + latentSize - 1) / latentSize;
    }

    public static float[] Mask(long length, int total)
    {
        var mask = new float[total];
        for (int i = 0; i < total; i++)
            mask[i] = i < length ? 1f : 0f;
        return mask;
    }

    public static List<string> ChunkText(string text, int maxLen = 300)
    {
        var chunks = new List<string>();

        var paragraphs = Regex.Split(text.Trim(), @"\n\s*\n+")
            .Select(p => p.Trim())
            .Where(p => !string.IsNullOrEmpty(p))
            .ToList();

        var sentenceRegex = new Regex(
            @"(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+");

        foreach (var paragraph in paragraphs)
        {
            var sentences = sentenceRegex.Split(paragraph);
            string currentChunk = "";

            foreach (var sentence in sentences)
            {
                if (string.IsNullOrEmpty(sentence)) continue;

                if (currentChunk.Length + sentence.Length + 1 <= maxLen)
                {
                    if (!string.IsNullOrEmpty(currentChunk)) currentChunk += " ";
                    currentChunk += sentence;
                }
                else
                {
                    if (!string.IsNullOrEmpty(currentChunk)) chunks.Add(currentChunk.Trim());
                    currentChunk = sentence;
                }
            }

            if (!string.IsNullOrEmpty(currentChunk)) chunks.Add(currentChunk.Trim());
        }

        if (chunks.Count == 0) chunks.Add(text.Trim());
        return chunks;
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter FullyQualifiedName~SupertonicHelpersTests`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Tts/SupertonicHelpers.cs Converse.Api.Tests/SupertonicHelpersTests.cs
git commit -m "Add SupertonicHelpers (mask/latent math + text chunking)"
```

---

## Task 6: Extend `SupertonicOptions`; drop `CfgScale`

**Files:**
- Modify: `Converse.Api/Configuration/SupertonicOptions.cs`
- Modify: `Converse.Api/appsettings.json`
- Modify: `Converse.Api/Tts/SupertonicPipeline.cs` (remove `CfgScale` from the log line only)

- [ ] **Step 1: Update `SupertonicOptions`**

Replace the body of `Converse.Api/Configuration/SupertonicOptions.cs` with:
```csharp
namespace Converse.Api.Configuration;

public sealed class SupertonicOptions
{
    public const string SectionName = "Supertonic";

    public string ModelsDirectory { get; set; } = "models/supertonic";

    // Directory holding voice-style JSON files (e.g. M1.json).
    public string VoicesDirectory { get; set; } = "models/supertonic/voices";

    // Default voice-style file name (without extension).
    public string DefaultVoice { get; set; } = "M1";

    // Language code for synthesis (wrapped as <lang>…</lang>). German by default.
    public string Language { get; set; } = "de";

    // Flow-matching denoising step count (passed to the model as total_step).
    public int CfmSteps { get; set; } = 16;

    // Speech speed factor; predicted duration is divided by this. 1.05 matches the reference default.
    public float Speed { get; set; } = 1.05f;
}
```

- [ ] **Step 2: Update `appsettings.json`**

In `Converse.Api/appsettings.json`, replace the `"Supertonic"` block with:
```json
  "Supertonic": {
    "ModelsDirectory": "models/supertonic",
    "VoicesDirectory": "models/supertonic/voices",
    "DefaultVoice": "M1",
    "Language": "de",
    "CfmSteps": 16,
    "Speed": 1.05
  },
```

- [ ] **Step 3: Fix the `SupertonicPipeline` startup log line**

In `Converse.Api/Tts/SupertonicPipeline.cs`, the existing `_logger.LogInformation("Supertonic loaded: ...")` call references `_opts.CfgScale`. Remove `cfg_scale={Cfg}` from the template and the `_opts.CfgScale` argument so it compiles. (Leave `cfm_steps={Steps}` / `_opts.CfmSteps`.)

- [ ] **Step 4: Build to verify it compiles**

Run: `dotnet build -clp:ErrorsOnly`
Expected: Build succeeded, 0 errors.

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Configuration/SupertonicOptions.cs Converse.Api/appsettings.json Converse.Api/Tts/SupertonicPipeline.cs
git commit -m "Add voice/language/speed options; remove unused CfgScale"
```

---

## Task 7: Load the default voice + processor in `SupertonicPipeline`

**Files:**
- Modify: `Converse.Api/Tts/SupertonicPipeline.cs`

- [ ] **Step 1: Add fields and load the voice/processor in the constructor**

In `SupertonicPipeline`, add fields near the other readonly fields:
```csharp
    private readonly VoiceStyle? _voice;
    private readonly SupertonicTextProcessor? _processor;
```
Inside the constructor's `try` block, AFTER `Indexer = UnicodeIndexer.Load(indexerPath);` and the 4 `InferenceSession` assignments, add the voice + processor load. Note the existing `IsReady = true;` line at the end of the `try` — gate it on the voice being present:
```csharp
            var voicePath = Path.Combine(Path.GetFullPath(_opts.VoicesDirectory), _opts.DefaultVoice + ".json");
            if (!File.Exists(voicePath))
                throw new FileNotFoundException($"Required Supertonic voice file missing: {voicePath}");
            _voice = VoiceStyle.Load(voicePath);
            _processor = new SupertonicTextProcessor(Indexer);
```
(The existing `catch` already sets `IsReady = false` and logs, so a missing/bad voice file correctly makes the pipeline not-ready.)

- [ ] **Step 2: Build to verify it compiles**

Run: `dotnet build -clp:ErrorsOnly`
Expected: Build succeeded. (`_voice`/`_processor` are unused for now — that is fine; they're used in Task 8.)

- [ ] **Step 3: Commit**

```bash
git add Converse.Api/Tts/SupertonicPipeline.cs
git commit -m "Load default voice style and text processor at pipeline startup"
```

---

## Task 8: Implement `SupertonicPipeline.Synthesize`

**Files:**
- Modify: `Converse.Api/Tts/SupertonicPipeline.cs`

- [ ] **Step 1: Replace the `Synthesize` scaffold with the real implementation**

In `Converse.Api/Tts/SupertonicPipeline.cs`, add `using Microsoft.ML.OnnxRuntime.Tensors;` at the top if not present. Replace the entire `Synthesize` method (the one that throws `NotImplementedException`) with:

```csharp
    public float[] Synthesize(string text, CancellationToken ct)
    {
        if (!IsReady || _processor is null || _voice is null || Config is null)
            throw new InvalidOperationException(
                "Supertonic pipeline is not ready; check models/voice directory configuration and startup logs.");

        int maxLen = (_opts.Language is "ko" or "ja") ? 120 : 300;
        var chunks = SupertonicHelpers.ChunkText(text, maxLen);

        const float silenceSeconds = 0.3f;
        var output = new List<float>();
        foreach (var chunk in chunks)
        {
            ct.ThrowIfCancellationRequested();
            var wav = InferChunk(chunk, ct);
            if (output.Count > 0)
                output.AddRange(new float[(int)(silenceSeconds * Config.SampleRate)]);
            output.AddRange(wav);
        }
        return output.ToArray();
    }

    private float[] InferChunk(string chunk, CancellationToken ct)
    {
        var (textIds, textMask) = _processor!.Encode(chunk, _opts.Language);
        int seq = textIds.Length;

        var textIdsTensor = new DenseTensor<long>(textIds, new[] { 1, seq });
        var textMaskTensor = new DenseTensor<float>(textMask, new[] { 1, 1, seq });
        var styleTtl = new DenseTensor<float>(_voice!.Ttl, _voice.TtlShape.Select(x => (int)x).ToArray());
        var styleDp = new DenseTensor<float>(_voice.Dp, _voice.DpShape.Select(x => (int)x).ToArray());

        // 1) Duration predictor -> one total-duration value per utterance.
        float[] duration;
        using (var dpOut = _durationPredictor!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
            NamedOnnxValue.CreateFromTensor("style_dp", styleDp),
            NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
        }))
        {
            duration = dpOut.First(o => o.Name == "duration").AsTensor<float>().ToArray();
        }
        for (int i = 0; i < duration.Length; i++)
            duration[i] /= _opts.Speed;

        // 2) Text encoder -> text_emb (copied out so we can dispose the run results).
        DenseTensor<float> textEmb;
        using (var teOut = _textEncoder!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
            NamedOnnxValue.CreateFromTensor("style_ttl", styleTtl),
            NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
        }))
        {
            var t = teOut.First(o => o.Name == "text_emb").AsTensor<float>();
            textEmb = new DenseTensor<float>(t.ToArray(), t.Dimensions.ToArray());
        }

        // 3) Sample noisy latent + latent mask.
        float maxDuration = duration.Max();
        int latentLen = SupertonicHelpers.ComputeLatentLen(
            maxDuration, Config!.SampleRate, Config.BaseChunkSize, Config.ChunkCompressFactor);
        int latentDim = Config.LatentDim * Config.ChunkCompressFactor;

        long wavLength = (long)(maxDuration * Config.SampleRate);
        long latentLength = SupertonicHelpers.LatentLengthFor(
            wavLength, Config.BaseChunkSize, Config.ChunkCompressFactor);
        var latentMask = SupertonicHelpers.Mask(latentLength, latentLen);

        var xt = new float[latentDim * latentLen];
        var rng = new Random();
        for (int d = 0; d < latentDim; d++)
            for (int t = 0; t < latentLen; t++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                float sample = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                xt[d * latentLen + t] = sample * latentMask[t];
            }

        var latentMaskTensor = new DenseTensor<float>(latentMask, new[] { 1, 1, latentLen });
        int totalStep = _opts.CfmSteps;
        var totalStepTensor = new DenseTensor<float>(new[] { (float)totalStep }, new[] { 1 });

        // 4) Iterative denoising (no CFG).
        for (int step = 0; step < totalStep; step++)
        {
            ct.ThrowIfCancellationRequested();
            var noisy = new DenseTensor<float>(xt, new[] { 1, latentDim, latentLen });
            var currentStepTensor = new DenseTensor<float>(new[] { (float)step }, new[] { 1 });

            using var veOut = _vectorEstimator!.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("noisy_latent", noisy),
                NamedOnnxValue.CreateFromTensor("text_emb", textEmb),
                NamedOnnxValue.CreateFromTensor("style_ttl", styleTtl),
                NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
                NamedOnnxValue.CreateFromTensor("latent_mask", latentMaskTensor),
                NamedOnnxValue.CreateFromTensor("total_step", totalStepTensor),
                NamedOnnxValue.CreateFromTensor("current_step", currentStepTensor),
            });
            var denoised = veOut.First(o => o.Name == "denoised_latent").AsTensor<float>();
            var flat = denoised.ToArray();
            Array.Copy(flat, xt, xt.Length);
        }

        // 5) Vocoder -> waveform.
        var latentTensor = new DenseTensor<float>(xt, new[] { 1, latentDim, latentLen });
        using var vocOut = _vocoder!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("latent", latentTensor),
        });
        return vocOut.First(o => o.Name == "wav_tts").AsTensor<float>().ToArray();
    }
```

Note: `Synthesize` and `InferChunk` reference the existing private session fields (`_durationPredictor`, `_textEncoder`, `_vectorEstimator`, `_vocoder`) and `_opts`, `Config`. Confirm those field names match the file; adjust the references if the existing names differ.

- [ ] **Step 2: Build to verify it compiles**

Run: `dotnet build -clp:ErrorsOnly`
Expected: Build succeeded, 0 errors.

- [ ] **Step 3: Run the full existing suite to confirm no regressions**

Run: `dotnet test`
Expected: PASS (all prior tests still green).

- [ ] **Step 4: Commit**

```bash
git add Converse.Api/Tts/SupertonicPipeline.cs
git commit -m "Implement Supertonic 4-stage synthesis pipeline"
```

---

## Task 9: Gated integration smoke test

**Files:**
- Create: `Converse.Api.Tests/SupertonicPipelineSmokeTests.cs`

- [ ] **Step 1: Write the integration test**

Create `Converse.Api.Tests/SupertonicPipelineSmokeTests.cs`:
```csharp
using FluentAssertions;
using Converse.Api.Configuration;
using Converse.Api.Tts;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace Converse.Api.Tests;

public class SupertonicPipelineSmokeTests
{
    // Walks up from the test output dir to find the repo's models/supertonic folder.
    private static string? FindModelsDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "supertonic", "tts.json");
            if (File.Exists(candidate)) return Path.Combine(dir.FullName, "models", "supertonic");
            dir = dir.Parent;
        }
        return null;
    }

    [Fact]
    public void Synthesize_produces_valid_german_audio()
    {
        var modelsDir = FindModelsDir();
        if (modelsDir is null || !File.Exists(Path.Combine(modelsDir, "voices", "M1.json")))
            return; // Models/voices not present in this environment — skip.

        var opts = Options.Create(new SupertonicOptions
        {
            ModelsDirectory = modelsDir,
            VoicesDirectory = Path.Combine(modelsDir, "voices"),
            DefaultVoice = "M1",
            Language = "de",
            CfmSteps = 16,
            Speed = 1.05f,
        });

        using var pipeline = new SupertonicPipeline(opts, NullLogger<SupertonicPipeline>.Instance);
        pipeline.IsReady.Should().BeTrue();

        var samples = pipeline.Synthesize("Hallo, wie geht es dir?", CancellationToken.None);

        samples.Should().NotBeEmpty();
        samples.Should().OnlyContain(s => !float.IsNaN(s) && !float.IsInfinity(s));
        samples.Should().OnlyContain(s => s >= -1f && s <= 1f);
        // At least ~0.3s of audio at 44.1 kHz for a short sentence.
        samples.Length.Should().BeGreaterThan((int)(0.3 * pipeline.SampleRate));
    }
}
```

- [ ] **Step 2: Run the test**

Run: `dotnet test --filter FullyQualifiedName~SupertonicPipelineSmokeTests`
Expected: PASS (runs the real pipeline, since models + M1.json are present locally). If it instead crashes with an ONNX shape/name error, STOP and reconcile against `.refsrc/Helper.cs` and the Task 0 metadata log.

- [ ] **Step 3: Commit**

```bash
git add Converse.Api.Tests/SupertonicPipelineSmokeTests.cs
git commit -m "Add gated Supertonic synthesis smoke test"
```

---

## Task 10: End-to-end manual verification (listen)

**Files:** none (manual).

- [ ] **Step 1: Restart the app and synthesize German via the HTTP endpoint**

```bash
cd "C:/LOCAL FILES/Claude Code/DotNet/Converse" && ASPNETCORE_ENVIRONMENT=Development dotnet run --project Converse.Api > .refsrc/run.log 2>&1 &
sleep 12
curl -s http://127.0.0.1:5000/health
curl -s -X POST http://127.0.0.1:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Guten Tag! Schön, dich kennenzulernen. Wie geht es dir heute?"}' \
  --output .refsrc/german.wav
echo "wav bytes: $(wc -c < .refsrc/german.wav)"
```
Expected: `/health` shows `tts:true`; `german.wav` is a non-trivial WAV (tens of KB+).

- [ ] **Step 2: Listen and confirm**

Play `.refsrc/german.wav`. Confirm it is intelligible German in the M1 voice with `ö`/`ü`/`ß` pronounced correctly. Stop the app afterward. (No commit — verification only.)

---

## Task 11: Per-request voice/lang overrides (multi-voice)

**Files:**
- Modify: `Converse.Api/Tts/ITextToSpeechService.cs`
- Modify: `Converse.Api/Tts/SupertonicTextToSpeechService.cs`
- Modify: `Converse.Api/Tts/SupertonicPipeline.cs`
- Modify: `Converse.Api/Endpoints/TtsEndpoints.cs`
- Test: `Converse.Api.Tests/SupertonicPipelineSmokeTests.cs`

- [ ] **Step 1: Load ALL voices at startup (replace the single-voice load from Task 7)**

In `SupertonicPipeline`, replace the field `private readonly VoiceStyle? _voice;` with:
```csharp
    private readonly Dictionary<string, VoiceStyle> _voices = new();
```
In the constructor, replace the Task 7 voice-loading block (the `voicePath`/`_voice = VoiceStyle.Load(...)` lines) with:
```csharp
            var voicesDir = Path.GetFullPath(_opts.VoicesDirectory);
            foreach (var file in Directory.EnumerateFiles(voicesDir, "*.json"))
                _voices[Path.GetFileNameWithoutExtension(file)] = VoiceStyle.Load(file);
            if (!_voices.ContainsKey(_opts.DefaultVoice))
                throw new FileNotFoundException(
                    $"Default Supertonic voice '{_opts.DefaultVoice}' not found in {voicesDir}");
            _processor = new SupertonicTextProcessor(Indexer);
```
(`Directory.EnumerateFiles` throws if the directory is missing — caught by the existing `catch`, so the pipeline becomes not-ready, as before.)

- [ ] **Step 2: Change `Synthesize`/`InferChunk` to take voice + lang (replace the Task 8 versions)**

Replace the `Synthesize` and `InferChunk` methods from Task 8 with these. Only the signatures, voice/lang resolution, and the `styleTtl`/`styleDp`/`Encode` references change; the duration/encoder/CFM/vocoder body is identical to Task 8:
```csharp
    public float[] Synthesize(string text, string? voice, string? lang, CancellationToken ct)
    {
        if (!IsReady || _processor is null || Config is null)
            throw new InvalidOperationException(
                "Supertonic pipeline is not ready; check models/voice directory configuration and startup logs.");

        var voiceName = string.IsNullOrWhiteSpace(voice) ? _opts.DefaultVoice : voice!;
        if (!_voices.TryGetValue(voiceName, out var style))
            throw new ArgumentException(
                $"Unknown voice '{voiceName}'. Available: {string.Join(", ", _voices.Keys.OrderBy(k => k))}.");

        var language = string.IsNullOrWhiteSpace(lang) ? _opts.Language : lang!;

        int maxLen = (language is "ko" or "ja") ? 120 : 300;
        var chunks = SupertonicHelpers.ChunkText(text, maxLen);

        const float silenceSeconds = 0.3f;
        var output = new List<float>();
        foreach (var chunk in chunks)
        {
            ct.ThrowIfCancellationRequested();
            var wav = InferChunk(chunk, style, language, ct);
            if (output.Count > 0)
                output.AddRange(new float[(int)(silenceSeconds * Config.SampleRate)]);
            output.AddRange(wav);
        }
        return output.ToArray();
    }

    private float[] InferChunk(string chunk, VoiceStyle voice, string lang, CancellationToken ct)
    {
        var (textIds, textMask) = _processor!.Encode(chunk, lang);
        int seq = textIds.Length;

        var textIdsTensor = new DenseTensor<long>(textIds, new[] { 1, seq });
        var textMaskTensor = new DenseTensor<float>(textMask, new[] { 1, 1, seq });
        var styleTtl = new DenseTensor<float>(voice.Ttl, voice.TtlShape.Select(x => (int)x).ToArray());
        var styleDp = new DenseTensor<float>(voice.Dp, voice.DpShape.Select(x => (int)x).ToArray());

        // 1) Duration predictor -> one total-duration value per utterance.
        float[] duration;
        using (var dpOut = _durationPredictor!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
            NamedOnnxValue.CreateFromTensor("style_dp", styleDp),
            NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
        }))
        {
            duration = dpOut.First(o => o.Name == "duration").AsTensor<float>().ToArray();
        }
        for (int i = 0; i < duration.Length; i++)
            duration[i] /= _opts.Speed;

        // 2) Text encoder -> text_emb (copied out so we can dispose the run results).
        DenseTensor<float> textEmb;
        using (var teOut = _textEncoder!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("text_ids", textIdsTensor),
            NamedOnnxValue.CreateFromTensor("style_ttl", styleTtl),
            NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
        }))
        {
            var t = teOut.First(o => o.Name == "text_emb").AsTensor<float>();
            textEmb = new DenseTensor<float>(t.ToArray(), t.Dimensions.ToArray());
        }

        // 3) Sample noisy latent + latent mask.
        float maxDuration = duration.Max();
        int latentLen = SupertonicHelpers.ComputeLatentLen(
            maxDuration, Config!.SampleRate, Config.BaseChunkSize, Config.ChunkCompressFactor);
        int latentDim = Config.LatentDim * Config.ChunkCompressFactor;

        long wavLength = (long)(maxDuration * Config.SampleRate);
        long latentLength = SupertonicHelpers.LatentLengthFor(
            wavLength, Config.BaseChunkSize, Config.ChunkCompressFactor);
        var latentMask = SupertonicHelpers.Mask(latentLength, latentLen);

        var xt = new float[latentDim * latentLen];
        var rng = new Random();
        for (int d = 0; d < latentDim; d++)
            for (int t = 0; t < latentLen; t++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                float sample = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                xt[d * latentLen + t] = sample * latentMask[t];
            }

        var latentMaskTensor = new DenseTensor<float>(latentMask, new[] { 1, 1, latentLen });
        int totalStep = _opts.CfmSteps;
        var totalStepTensor = new DenseTensor<float>(new[] { (float)totalStep }, new[] { 1 });

        // 4) Iterative denoising (no CFG).
        for (int step = 0; step < totalStep; step++)
        {
            ct.ThrowIfCancellationRequested();
            var noisy = new DenseTensor<float>(xt, new[] { 1, latentDim, latentLen });
            var currentStepTensor = new DenseTensor<float>(new[] { (float)step }, new[] { 1 });

            using var veOut = _vectorEstimator!.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("noisy_latent", noisy),
                NamedOnnxValue.CreateFromTensor("text_emb", textEmb),
                NamedOnnxValue.CreateFromTensor("style_ttl", styleTtl),
                NamedOnnxValue.CreateFromTensor("text_mask", textMaskTensor),
                NamedOnnxValue.CreateFromTensor("latent_mask", latentMaskTensor),
                NamedOnnxValue.CreateFromTensor("total_step", totalStepTensor),
                NamedOnnxValue.CreateFromTensor("current_step", currentStepTensor),
            });
            var denoised = veOut.First(o => o.Name == "denoised_latent").AsTensor<float>();
            var flat = denoised.ToArray();
            Array.Copy(flat, xt, xt.Length);
        }

        // 5) Vocoder -> waveform.
        var latentTensor = new DenseTensor<float>(xt, new[] { 1, latentDim, latentLen });
        using var vocOut = _vocoder!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("latent", latentTensor),
        });
        return vocOut.First(o => o.Name == "wav_tts").AsTensor<float>().ToArray();
    }
```

- [ ] **Step 3: Add the 4-arg overload to the interface and service**

In `Converse.Api/Tts/ITextToSpeechService.cs`, add below the existing `SynthesizeAsync`:
```csharp
    Task<float[]> SynthesizeAsync(string text, string? voice, string? lang, CancellationToken ct);
```
Replace `Converse.Api/Tts/SupertonicTextToSpeechService.cs`'s single `SynthesizeAsync` method with both overloads:
```csharp
    public Task<float[]> SynthesizeAsync(string text, CancellationToken ct)
        => SynthesizeAsync(text, null, null, ct);

    public Task<float[]> SynthesizeAsync(string text, string? voice, string? lang, CancellationToken ct)
        => Task.FromResult(_pipeline.Synthesize(text, voice, lang, ct));
```
(The conversation `/turn` path keeps calling the 2-arg overload, so it is unaffected.)

- [ ] **Step 4: Accept `voice`/`lang` on the `/tts` endpoint and map bad input to 400**

In `Converse.Api/Endpoints/TtsEndpoints.cs`, replace the `/tts` handler body and the `TtsRequest` record:
```csharp
        app.MapPost("/tts", async (
            TtsRequest req,
            ITextToSpeechService tts,
            IAudioConverter audio,
            CancellationToken ct) =>
        {
            if (!tts.IsReady)
                return Results.Problem("Supertonic TTS is not ready; check model path configuration.", statusCode: 503);

            float[] samples;
            try
            {
                samples = await tts.SynthesizeAsync(req.Text, req.Voice, req.Lang, ct);
            }
            catch (ArgumentException ex)
            {
                return Results.Problem(ex.Message, statusCode: 400);
            }

            var bytes = audio.PcmToWav(samples, tts.SampleRate);
            return Results.File(bytes, "audio/wav");
        });
```
and:
```csharp
internal sealed record TtsRequest(string Text, string? Voice = null, string? Lang = null);
```

- [ ] **Step 5: Add voice-override cases to the gated smoke test**

Append these members to the `SupertonicPipelineSmokeTests` class in `Converse.Api.Tests/SupertonicPipelineSmokeTests.cs` (`FindModelsDir` already exists from Task 9):
```csharp
    private static SupertonicPipeline? TryBuildPipeline()
    {
        var modelsDir = FindModelsDir();
        if (modelsDir is null || !File.Exists(Path.Combine(modelsDir, "voices", "M1.json")))
            return null;
        var opts = Options.Create(new SupertonicOptions
        {
            ModelsDirectory = modelsDir,
            VoicesDirectory = Path.Combine(modelsDir, "voices"),
            DefaultVoice = "M1",
            Language = "de",
            CfmSteps = 16,
            Speed = 1.05f,
        });
        return new SupertonicPipeline(opts, NullLogger<SupertonicPipeline>.Instance);
    }

    [Fact]
    public void Synthesize_with_non_default_voice_produces_audio()
    {
        using var pipeline = TryBuildPipeline();
        if (pipeline is null) return; // models not present — skip
        var samples = pipeline.Synthesize("Guten Morgen.", "F1", "de", CancellationToken.None);
        samples.Should().NotBeEmpty();
    }

    [Fact]
    public void Synthesize_with_unknown_voice_throws_argument_exception()
    {
        using var pipeline = TryBuildPipeline();
        if (pipeline is null) return; // models not present — skip
        var act = () => pipeline.Synthesize("Hallo.", "NopeVoice", "de", CancellationToken.None);
        act.Should().Throw<ArgumentException>();
    }
```

- [ ] **Step 6: Build and run tests**

Run: `dotnet build -clp:ErrorsOnly` then `dotnet test`
Expected: Build succeeded; all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Converse.Api/Tts/ITextToSpeechService.cs Converse.Api/Tts/SupertonicTextToSpeechService.cs Converse.Api/Tts/SupertonicPipeline.cs Converse.Api/Endpoints/TtsEndpoints.cs Converse.Api.Tests/SupertonicPipelineSmokeTests.cs
git commit -m "Support per-request voice/lang overrides on /tts"
```

---

## Task 12: Enable CORS for browser/extension access

**Files:**
- Modify: `Converse.Api/Program.cs`
- Test: `Converse.Api.Tests/CorsTests.cs`

- [ ] **Step 1: Write the failing test**

Create `Converse.Api.Tests/CorsTests.cs`:
```csharp
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Converse.Api.Tests;

public class CorsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    public CorsTests(WebApplicationFactory<Program> factory) => _factory = factory;

    [Fact]
    public async Task Cross_origin_request_gets_allow_origin_header()
    {
        var client = _factory.CreateClient();
        var req = new HttpRequestMessage(HttpMethod.Get, "/health");
        req.Headers.Add("Origin", "chrome-extension://abcdefghijklmnop");

        var resp = await client.SendAsync(req);

        resp.Headers.Contains("Access-Control-Allow-Origin").Should().BeTrue();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter FullyQualifiedName~CorsTests`
Expected: FAIL — no `Access-Control-Allow-Origin` header (CORS not configured).

- [ ] **Step 3: Add CORS to `Program.cs`**

In `Converse.Api/Program.cs`, after the other `builder.Services` registrations (before `var app = builder.Build();`), add:
```csharp
// CORS — allow browser/extension callers (local-only API; tighten origins later).
builder.Services.AddCors(options => options.AddDefaultPolicy(policy =>
    policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));
```
After `var app = builder.Build();` and before the first `app.Map...` call, add:
```csharp
app.UseCors();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter FullyQualifiedName~CorsTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Converse.Api/Program.cs Converse.Api.Tests/CorsTests.cs
git commit -m "Enable CORS so the Chrome extension can call the API"
```

---

## Task 13: Final verification, cleanup, and push

**Files:** none tracked (manual + cleanup).

- [ ] **Step 1: Restart the app and verify overrides + CORS over HTTP**

```bash
cd "C:/LOCAL FILES/Claude Code/DotNet/Converse" && ASPNETCORE_ENVIRONMENT=Development dotnet run --project Converse.Api > .refsrc/run.log 2>&1 &
sleep 12
# default voice
curl -s -X POST http://127.0.0.1:5000/tts -H "Content-Type: application/json" \
  -d '{"text":"Guten Tag! Wie geht es dir?"}' --output .refsrc/default.wav
# voice override
curl -s -X POST http://127.0.0.1:5000/tts -H "Content-Type: application/json" \
  -d '{"text":"Guten Tag!","voice":"F1"}' --output .refsrc/f1.wav
# unknown voice -> expect HTTP 400
curl -s -o /dev/null -w "unknown-voice HTTP %{http_code}\n" -X POST http://127.0.0.1:5000/tts \
  -H "Content-Type: application/json" -d '{"text":"Hallo","voice":"Nope"}'
# CORS header present with an Origin
curl -s -D - -o /dev/null -H "Origin: chrome-extension://abc" http://127.0.0.1:5000/health | grep -i "access-control-allow-origin"
```
Expected: `default.wav` and `f1.wav` are non-trivial WAVs (and audibly different voices); the unknown-voice line prints `HTTP 400`; the CORS line shows `Access-Control-Allow-Origin`. Stop the app afterward.

- [ ] **Step 2: Remove the scratch directory**

```bash
cd "C:/LOCAL FILES/Claude Code/DotNet/Converse" && rm -rf .refsrc
```

- [ ] **Step 3: Confirm a clean tree and full green suite**

Run: `git status --short` (expect no `.refsrc` entries) and `dotnet test` (expect all PASS).

- [ ] **Step 4: Push**

```bash
git push
```

---

## Self-review

**Spec coverage:**
- 5-stage synthesis (tokenize/duration/encode/CFM/vocode) → Tasks 3, 5, 7, 8. ✓
- Exact tensor names → Task 8 (+ verified in Task 0). ✓
- `VoiceStyle` loader → Task 4. ✓
- `SupertonicTextProcessor` (NFKD, cleanup, lang wrap) → Task 3. ✓
- Config additions, `CfgScale` removal → Task 6. ✓
- Voice assets download (supertonic-3) → Task 0. ✓
- German correctness (umlaut/ß decomposition) → Task 3 tests + Task 10 listen. ✓
- Error handling → not-ready on missing default voice (Task 7/11); `400` on unknown voice / invalid lang → Task 11. ✓
- Tests: text processor, voice loader, math, integration, overrides, CORS → Tasks 3, 4, 5, 9, 11, 12. ✓
- Per-request voice/lang overrides on `/tts` (defaults when omitted; conversation path unaffected) → Task 11. ✓
- CORS for browser/extension access → Task 12. ✓
- Success criteria (German WAV, overrides honoured, cross-origin reachable, `tts:true`) → Tasks 10, 13. ✓

**Deviations from spec, intentional:**
- The noisy-latent RNG is not seeded. The smoke test asserts properties (finite, in range, plausible length), not exact samples, so determinism is unnecessary (YAGNI). Noted here so it isn't mistaken for an omission.
- Mask/length helpers are specialized to batch size 1 (our only path) rather than the reference's general batched `float[][][]`, keeping the code simpler. Faithful for single-utterance synthesis.

**Placeholder scan:** none — every code step contains complete code.

**Type consistency:** `SupertonicTextProcessor(UnicodeIndexer)`, `UnicodeIndexer.MapChar(int)→long`, `VoiceStyle.{Ttl,TtlShape,Dp,DpShape}`, `TtsConfig.{SampleRate,BaseChunkSize,LatentDim,ChunkCompressFactor}`, `SupertonicHelpers.{ComputeLatentLen,LatentLengthFor,Mask,ChunkText}`, and `SupertonicOptions.{Language,DefaultVoice,VoicesDirectory,CfmSteps,Speed}` are used consistently across Tasks 1–13. Note: `SupertonicPipeline.Synthesize` gains its final `(text, voice?, lang?, ct)` signature in Task 11 (Task 8 introduces the 2-arg form first); the `_voice` field from Task 7 becomes the `_voices` dictionary in Task 11. These are intentional incremental refinements — each task leaves the build green.
