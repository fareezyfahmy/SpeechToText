import whisper
import pydub
import os
from pydub import AudioSegment

# Load the Whisper model (small, medium, large depending on accuracy and speed)
model = whisper.load_model("small")

def transcribe_audio(audio_file):
    # Load the audio file
    audio = whisper.load_audio(audio_file)
    # Preprocess and convert the audio to 30 kHz
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Perform the transcription
    options = whisper.DecodingOptions(language="en")
    result = whisper.decode(model, mel, options)

    # Output the text
    print("Transcription: ", result.text)

# Convert mp3 to wav if necessary
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

# Example usage
if __name__ == "__main__":
    # Provide the path to your audio file (either .mp3 or .wav)
    input_audio_file = "path_to_your_audio_file.wav"  # or .mp3
    if input_audio_file.endswith(".mp3"):
        input_audio_file = convert_mp3_to_wav(input_audio_file)
    
    transcribe_audio(input_audio_file)
