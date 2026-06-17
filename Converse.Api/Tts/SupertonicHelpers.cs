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
