from gtts import gTTS
from sample_questions import SAMPLE_QUESTIONS
import os

def create_audio_questions():
    """Create audio versions of all sample questions"""
    
    # Create input directory
    os.makedirs('../data/audio/input', exist_ok=True)
    
    print("Creating audio question files...")
    
    for lang_code, questions in SAMPLE_QUESTIONS.items():
        print(f"\n=== Creating {lang_code.upper()} audio questions ===")
        
        for i, question in enumerate(questions):
            try:
                # Create audio file
                tts = gTTS(text=question, lang=lang_code)
                audio_path = f'../data/audio/input/question_{lang_code}_{i+1}.mp3'
                tts.save(audio_path)
                
                print(f"Created: question_{lang_code}_{i+1}.mp3")
                print(f"   Text: {question[:50]}...")
                
            except Exception as e:
                print(f"Failed to create question_{lang_code}_{i+1}.mp3: {e}")
    
    print("\n Audio question files created successfully!")

if __name__ == "__main__":
    create_audio_questions()
