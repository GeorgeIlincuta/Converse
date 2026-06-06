using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Converse.Api.Audio;

public sealed class NAudioConverter : IAudioConverter
{
    public float[] ToWhisperPcm(Stream audio)
    {
        using var buffer = new MemoryStream();
        audio.CopyTo(buffer);
        buffer.Position = 0;

        WaveStream waveStream;
        try
        {
            waveStream = new WaveFileReader(buffer);
        }
        catch (Exception ex) when (ex is FormatException or InvalidOperationException or ArgumentException)
        {
            buffer.Position = 0;
            try
            {
                waveStream = new StreamMediaFoundationReader(buffer);
            }
            catch (Exception mfEx)
            {
                throw new InvalidOperationException(
                    "Could not decode audio: no supported decoder for the input format.", mfEx);
            }
        }

        using (waveStream)
        {
            ISampleProvider samples = waveStream.ToSampleProvider();

            switch (waveStream.WaveFormat.Channels)
            {
                case 1: break;
                case 2: samples = new StereoToMonoSampleProvider(samples); break;
                default:
                    throw new InvalidOperationException(
                        $"Unsupported channel count {waveStream.WaveFormat.Channels}; only mono and stereo input are supported.");
            }

            if (waveStream.WaveFormat.SampleRate != 16000)
                samples = new WdlResamplingSampleProvider(samples, 16000);

            var result = new List<float>();
            var chunk = new float[4096];
            int read;
            while ((read = samples.Read(chunk, 0, chunk.Length)) > 0)
            {
                for (int i = 0; i < read; i++)
                    result.Add(chunk[i]);
            }
            return result.ToArray();
        }
    }

    public byte[] PcmToWav(float[] samples, int sampleRate)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be greater than zero.");

        int dataSize = samples.Length * 2;
        var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, System.Text.Encoding.ASCII, leaveOpen: true);

        writer.Write(new[] { 'R', 'I', 'F', 'F' });
        writer.Write((uint)(36 + dataSize));
        writer.Write(new[] { 'W', 'A', 'V', 'E' });
        writer.Write(new[] { 'f', 'm', 't', ' ' });
        writer.Write((uint)16);          // subchunk1 size
        writer.Write((ushort)1);         // PCM
        writer.Write((ushort)1);         // mono
        writer.Write((uint)sampleRate);
        writer.Write((uint)(sampleRate * 2)); // byte rate
        writer.Write((ushort)2);         // block align
        writer.Write((ushort)16);        // bits per sample
        writer.Write(new[] { 'd', 'a', 't', 'a' });
        writer.Write((uint)dataSize);

        foreach (var sample in samples)
        {
            // *32767 not *32768 to keep the range symmetric: max positive == 32767, max negative clamped to -32767
            var s = (short)(Math.Clamp(sample, -1f, 1f) * 32767f);
            writer.Write(s);
        }

        return ms.ToArray();
    }
}
