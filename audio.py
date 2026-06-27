import pyttsx3
import speech_recognition as sr

def speak_text(text):
    print("\n🔊 Speaking output...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.say(text)
    engine.runAndWait()

def listen_for_question():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Adjusting for background noise... Please remain silent for 1 second.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("🗣️ Speak your question now...")
        audio = recognizer.listen(source)

    print("⏳ Transcribing your speech...")

    try:
        question = recognizer.recognize_google(audio)
        return question
    except sr.UnknownValueError:
        print("❌ Sorry, I could not understand the audio. Defaulting to fallback.")
        return "If the order_cache_tier_1 Redis cluster goes down, who should be paged?"
    except sr.RequestError:
        print("❌ Could not connect to the STT service. Defaulting to fallback.")
        return "If the order_cache_tier_1 Redis cluster goes down, who should be paged?"