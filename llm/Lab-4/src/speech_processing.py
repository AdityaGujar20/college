import whisper
from gtts import gTTS
import pyttsx3
import os

class SpeechProcessor:
    def __init__(self):
        print("Loading Whisper model (this may take a few minutes on first run)...")
        self.whisper_model = whisper.load_model("base")
        
        try:
            self.offline_tts = pyttsx3.init()
        except:
            self.offline_tts = None
        
        print("Speech processing initialized successfully!")
    
    def speech_to_text(self, audio_file_path, language_code=None):
        """Convert speech to text using OpenAI Whisper (free)"""
        try:
            lang_map = {
                'en-US': 'en',
                'mr-IN': 'mr',  # Marathi
                'fr-FR': 'fr'   # French
            }
            
            whisper_lang = lang_map.get(language_code, None)
            
            print(f"Transcribing audio in {whisper_lang}...")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_file_path, 
                language=whisper_lang,
                verbose=False
            )
            
            transcribed_text = result["text"].strip()
            print(f"Transcription: {transcribed_text}")
            return transcribed_text
        
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return f"Could not transcribe audio: {e}"
    
    def text_to_speech(self, text, language_code, output_path):
        """Convert text to speech using multiple free methods"""
        
        # Try gTTS first (Google Text-to-Speech - free, no credentials)
        if self._try_gtts(text, language_code, output_path):
            return True
        
        # Fallback to offline TTS
        print("gTTS failed, trying offline TTS...")
        if self._try_offline_tts(text, output_path):
            return True
        
        print("All TTS methods failed")
        return False
    
    def _try_gtts(self, text, language_code, output_path):
        """Try Google Text-to-Speech (gTTS)"""
        try:
            # Map language codes
            lang_map = {
                'en-US': 'en',
                'mr-IN': 'mr',  # Marathi
                'fr-FR': 'fr'   # French
            }
            
            tts_lang = lang_map.get(language_code, 'en')
            
            print(f"Generating speech in {tts_lang} using gTTS...")
            
            # Generate speech
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.save(output_path)
            
            print(f"✓ Generated audio using gTTS: {output_path}")
            return True
        
        except Exception as e:
            print(f"gTTS failed: {e}")
            return False
    
    def _try_offline_tts(self, text, output_path):
        """Try offline TTS as backup"""
        try:
            if self.offline_tts is None:
                return False
            
            wav_path = output_path.replace('.mp3', '.wav')
            
            self.offline_tts.save_to_file(text, wav_path)
            self.offline_tts.runAndWait()
            
            print(f"Generated audio using offline TTS: {wav_path}")
            return True
        
        except Exception as e:
            print(f"Offline TTS failed: {e}")
            return False
