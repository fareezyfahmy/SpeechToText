import whisper
import pydub
import os
from pydub import AudioSegment
import torchaudio

# Load the Whisper model
model = whisper.load_model("small")

def transcribe_audio(audio_file):
    # Load the audio file using torchaudio
    audio, sr = torchaudio.load(audio_file)
    
    # Resample the audio to 30 kHz if it's not already
    if sr != 30000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=30000)
        audio = resampler(audio)
    
    # Normalize audio to -1 to 1 range
    audio = audio / audio.abs().max()

    # Pad or trim the audio to the required length for Whisper
    audio = whisper.pad_or_trim(audio)
    
    # Convert audio to Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Perform the transcription
    options = whisper.DecodingOptions(language="en")
    result = whisper.decode(model, mel, options)

    # Output the transcription text
    print("Transcription: ", result["text"])  # Fix applied here

    # Save the transcription to a .txt file
    output_file = audio_file.replace(".wav", ".txt").replace(".mp3", ".txt")
    with open(output_file, "w") as f:
        f.write(result["text"])  # Fix applied here

    print(f"Transcription saved to: {output_file}")

# MP3 to WAV conversion if needed
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

if __name__ == "__main__":
    # Provide the path to your audio file (mp3 or wav format)
    input_audio_file = "C:\\Users\\muhdf\\Downloads\\Hello hello good morning....mp3"
    if input_audio_file.endswith(".mp3"):
        input_audio_file = convert_mp3_to_wav(input_audio_file)
    
    transcribe_audio(input_audio_file)
