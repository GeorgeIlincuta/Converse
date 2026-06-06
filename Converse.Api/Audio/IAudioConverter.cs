namespace Converse.Api.Audio;

public interface IAudioConverter
{
    // Decode arbitrary input audio (WAV/mp3/etc.) and return mono 16 kHz float32 samples
    // in the range [-1, 1]. Throws InvalidOperationException with a clear message if the
    // input format can't be decoded.
    float[] ToWhisperPcm(Stream audio);

    // Wrap float32 PCM samples in a standard RIFF/WAVE container (16-bit PCM, mono),
    // returning the complete WAV file bytes.
    byte[] PcmToWav(float[] samples, int sampleRate);
}
