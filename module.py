import os
import openai
import time
from gtts import gTTS
from playsound import playsound
from dotenv import load_dotenv

import requests

load_dotenv()

class TextToSpeech:
    def __init__(self) -> None:
        pass

    def run_speech(self, text_to_speak):

        language = 'en'

        gtts_obj = gTTS(text=text_to_speak, lang=language, slow=False)

        audio_file = "text_to_speech.mp3"
        gtts_obj.save(audio_file)

        playsound(audio_file)

        os.remove(audio_file)

class TTSBry:
    def __init__(self, url):
        self.url = url

    def save_wav_file(self, response_content, filename):
        with open(filename, 'wb') as f:
            f.write(response_content)

    def play_wav_file(self, filename):
        playsound(filename)

    def delete_wav_file(self, filename):
        os.remove(filename)

    def make_request(self, payload):
        response = requests.post(self.url, json=payload)
        return response

    def speak_text(self, payload):
        response = self.make_request(payload)
        if response.status_code == 200:
            filename = 'output.wav'
            self.save_wav_file(response.content, filename)
            self.play_wav_file(filename)
            time.sleep(2)  # Adjust sleep time as needed
            self.delete_wav_file(filename)
            print("File played and deleted successfully.")
        else:
            print(f"Error: {response.status_code} - {response.text}")

"""
def main():
    url = 'YOUR_API_ENDPOINT_HERE'
    payload = {"text": "hello there!"}
    handler = ApiRequestHandler(url)
    handler.speak_text(payload)
"""

class LargeLanguageModelAPI:

    def __init__(self):
        self.openai_key = os.getenv('OPENAI_KEY')
    
    def run_gpt(self, text):

        openai.api_key = self.openai_key
        
        custom_prompt = """    
        You are a Scrum Master for SP Madrid. Your name is Carlo. The company has Software Developers. 
        The software developers are Bryner, Joshua, Julius, Kevin, Lamby, Jason, Johnmel, and Jerome.
        """
        
        
        prompt = "You are a helpful assistant"


        collection_prompt = """
"""
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.001,
        max_tokens=100
        )

        print(response['choices'][0]['message']['content'])

        return response['choices'][0]['message']['content']