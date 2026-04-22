# speech_to_text.py

from groq import Groq
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

client = Groq(api_key="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")

def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Recording...")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    wav.write(filename, fs, audio)
    print("Recording done")

    return filename


def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )

    return transcription.text