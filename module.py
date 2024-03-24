import os
from openai import OpenAI
from gtts import gTTS
from playsound import playsound
from dotenv import load_dotenv

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

class LargeLanguageModelAPI:

    def __init__(self):
        self.openai_key = os.getenv('OPENAI_KEY')
    
    def run_gpt(self, text):

        client = OpenAI(api_key=self.openai_key)

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an helpful assistant."},
            {"role": "user", "content": text}
        ]
        )

        print(completion.choices[0].message.content)

        return completion.choices[0].message.content