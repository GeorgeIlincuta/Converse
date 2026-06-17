using System.Text.Json;

namespace Converse.Api.Tts;

public sealed class UnicodeIndexer
{
    private readonly int[] _table;

    public int TableSize => _table.Length;

    public int VocabSize { get; }

    public UnicodeIndexer(int[] table)
    {
        _table = table ?? throw new ArgumentNullException(nameof(table));
        var max = -1;
        foreach (var v in table)
            if (v > max) max = v;
        VocabSize = max + 1;
    }

    public static UnicodeIndexer Load(string path)
    {
        using var stream = File.OpenRead(path);
        var table = JsonSerializer.Deserialize<int[]>(stream)
            ?? throw new InvalidDataException($"unicode_indexer.json at '{path}' did not deserialize to int[].");
        return new UnicodeIndexer(table);
    }

    // Maps a string to token ids by iterating Unicode scalar values and looking up
    // the per-codepoint index in the table. Codepoints with value -1 (or out of range)
    // are dropped — they're unsupported characters.
    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text)) return Array.Empty<int>();

        var tokens = new List<int>(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            var cp = rune.Value;
            if (cp >= 0 && cp < _table.Length)
            {
                var id = _table[cp];
                if (id >= 0) tokens.Add(id);
            }
        }
        return tokens.ToArray();
    }
}
