import librosa


def convert_to_16k_mono_librosa(input_file):
    # Load the audio file
    audio_data, sr = librosa.load(input_file, sr=None, mono=False)

    # Determine if the audio is mono or stereo
    if len(audio_data.shape) == 1:
        channels = "Mono"
    elif len(audio_data.shape) == 2:
        channels = "Stereo"
    else:
        raise ValueError("Unknown channel configuration")

    # Convert to mono if it's stereo
    if channels == "Stereo":
        audio_data = librosa.to_mono(audio_data)

    # Resample to 16 kHz if needed
    if sr != 16000:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=16000)  # THE NUMPY ARRAY THAT GOES TO TZUR
        sr = 16000

    return audio_data,sr

if __name__ == '__main__':
    input_wav = r"C:\Program Files (x86)\Steam\friends\friend_online.wav"
    output_wav = "out.wav"
    convert_to_16k_mono_librosa(input_wav, output_wav)
