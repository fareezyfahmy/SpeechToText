import whisper
import pydub
import os
import librosa
import numpy as np
from pydub import AudioSegment

# Load Whisper model - choose between small, medium, large
model = whisper.load_model("small")

# Function to resample and normalize audio
def preprocess_audio(audio_file, target_sr=30000):

    audio, sr = librosa.load(audio_file, sr=None)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    audio = librosa.util.normalize(audio)

    return audio, target_sr

def transcribe_audio(audio_file):
    audio, sr = preprocess_audio(audio_file)

    # audio = librosa.load(audio_file)

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # transcription
    options = whisper.DecodingOptions(language="en")
    result = whisper.decode(model, mel, options)

    print("Transcription: ", result.text)

    # Save transcription to a .txt file
    output_file = audio_file.replace(".wav", ".txt").replace(".mp3", ".txt")
    with open(output_file, "w") as f:
        f.write(result.text)

    print(f"Transcription saved to: {output_file}")

# mp3 to wav conversion
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

if __name__ == "__main__":
    # audio file path (mp3 or wav file format)
    input_audio_file = "C:\Users\muhdf\Documents\WORK\MAISTORAGE\Speech2Text\Hello hello good morning....wav"
    if input_audio_file.endswith(".mp3"):
        input_audio_file = convert_mp3_to_wav(input_audio_file)
    
    transcribe_audio(input_audio_file)
