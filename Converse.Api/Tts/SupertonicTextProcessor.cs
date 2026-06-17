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
        const string EndingChars = ".!?;:,'\"“”‘’')]}…。」』】〉》›»";
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
