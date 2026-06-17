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

    // Maps a single UTF-16 code unit (cast to int) to its token id, matching the
    // reference: in-range codepoints return the table value, everything else returns 0.
    public long MapChar(int charValue)
        => charValue >= 0 && charValue < _table.Length ? _table[charValue] : 0L;
}
