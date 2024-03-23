import os
import pyaudio
import wave
import webrtcvad
import collections
import time
import speech_recognition as sr
from module import TextToSpeech, LargeLanguageModelAPI

# Initialize TTS
tts = TextToSpeech()

# Initialize LLM
llm = LargeLanguageModelAPI()

# Initialize VAD
vad = webrtcvad.Vad(1)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
PADDING_DURATION_MS = 500
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = CHUNK_SIZE * 2
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)

# Initialize PyAudio
audio = pyaudio.PyAudio()

def save_speech(voiced_frames, sample_rate, output_filename):
    """Saves the voiced frames to a WAV file."""
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(voiced_frames))
    wf.close()
    print(f"Saved to {output_filename}")
    return output_filename  # Return the filename for further use

def transcribe_audio(file_path):
    """Transcribes the audio from a given file path."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Transcription: {text}")
            
            text_to_speak = llm.run_gpt(text)
            tts.run_speech(text_to_speak)
     
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")


        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    os.remove(file_path)

def create_recording_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE)

# Create a ring buffer to hold audio chunks for VAD
ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)

# Listen to the microphone
voiced_frames = []
triggered = False

print("Listening...")

while True:
    chunk = stream.read(CHUNK_SIZE)
    is_speech = vad.is_speech(chunk, RATE)
    
    if not triggered:
        ring_buffer.append(chunk)
        num_voiced = len([frame for frame in ring_buffer if vad.is_speech(frame, RATE)])
        
        if num_voiced > 0.9 * ring_buffer.maxlen:
            print("Starting phrase")
            triggered = True
            for rb_chunk in ring_buffer:
                voiced_frames.append(rb_chunk)
            ring_buffer.clear()
    else:
        voiced_frames.append(chunk)
        ring_buffer.append(chunk)
        
        num_unvoiced = len([frame for frame in ring_buffer if not vad.is_speech(frame, RATE)])
        
        if num_unvoiced > 0.9 * ring_buffer.maxlen:

            print("Ending phrase")

            triggered = False

            create_recording_dir('recording')

            output_filename = f"recording/audio_{int(time.time())}.wav"
            saved_file_path = save_speech(voiced_frames, RATE, output_filename)
            
            # Transcribe the saved speech
            transcribe_audio(saved_file_path)
            
            voiced_frames = []

