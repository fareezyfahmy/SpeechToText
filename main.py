import whisper
import pydub
import os
from pydub import AudioSegment

# Load model - choose between small, medium, large
model = whisper.load_model("small")

def transcribe_audio(audio_file):
    # Load audio file
    audio = whisper.load_audio(audio_file)
    # Preprocessing - convert audio to 30 kHz
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Transcription
    options = whisper.DecodingOptions(language="en")
    result = whisper.decode(model, mel, options)

    # Output - text
    print("Transcription: ", result.text)

# mp3 to wav conversion
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

if __name__ == "__main__":
    # Audio path (mp3 or wav file format)
    input_audio_file = "C:\\Users\\muhdf\\Downloads\\Hello hello good morning....mp3" 
    if input_audio_file.endswith(".mp3"):
        input_audio_file = convert_mp3_to_wav(input_audio_file)
    
    transcribe_audio(input_audio_file)
