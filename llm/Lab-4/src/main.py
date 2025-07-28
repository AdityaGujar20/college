from pdf_processing import extract_text_from_pdf, prepare_context
from speech_processing import SpeechProcessor
from qa_models import MultilingualQASystem
from sample_questions import SAMPLE_QUESTIONS, LANGUAGE_CODES
import os
import json

def main():
    # Initialize components
    print("Initializing system...")
    qa_system = MultilingualQASystem()
    speech_processor = SpeechProcessor()
    
    # PDF paths
    pdf_paths = {
        'en': '../data/pdfs/english.pdf',
        'mr': '../data/pdfs/marathi.pdf', 
        'fr': '../data/pdfs/french.pdf'
    }
    
    results = {}
    
    # Process each language
    for lang_code in ['en', 'mr', 'fr']:
        print(f"\n=== Processing {lang_code.upper()} ===")
        
        # Extract text from PDF
        context = extract_text_from_pdf(pdf_paths[lang_code])
        context = prepare_context(context)
        
        results[lang_code] = {
            'language': lang_code,
            'questions_answers': []
        }
        
        # Process each question with dual input support
        for i, question_text in enumerate(SAMPLE_QUESTIONS[lang_code]):
            print(f"\n--- Question {i+1} ---")
            print(f"Text version: {question_text}")
            
            # Check for audio input file
            audio_question_path = f"../data/audio/input/question_{lang_code}_{i+1}.mp3"
            
            question_source = "text"
            transcribed_question = None
            
            if os.path.exists(audio_question_path):
                try:
                    print("Audio file found - Using speech-to-text...")
                    transcribed_question = speech_processor.speech_to_text(
                        audio_question_path, 
                        LANGUAGE_CODES[lang_code]
                    )
                    print(f"Transcribed: {transcribed_question}")
                    
                    # Use transcribed question for QA
                    question_for_qa = transcribed_question
                    question_source = "audio"
                    
                except Exception as e:
                    print(f"Audio transcription failed: {e}")
                    print("Falling back to text input...")
                    question_for_qa = question_text
                    question_source = "text_fallback"
            else:
                print("No audio file found - Using text input...")
                question_for_qa = question_text
            
            print(f"Processing question: {question_for_qa}")
            
            # Generate answer
            answer = qa_system.answer_question(question_for_qa, context, lang_code)
            print(f"Answer: {answer}")
            
            # Convert answer to speech
            output_audio_path = f"../data/audio/output/answer_{lang_code}_{i+1}.mp3"
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            
            print("Generating speech output...")
            success = speech_processor.text_to_speech(
                answer, 
                LANGUAGE_CODES[lang_code], 
                output_audio_path
            )
            
            # Store comprehensive results
            results[lang_code]['questions_answers'].append({
                'question_text': question_text,
                'question_transcribed': transcribed_question,
                'question_used_for_qa': question_for_qa,
                'question_source': question_source,
                'audio_input_path': audio_question_path if os.path.exists(audio_question_path) else None,
                'answer': answer,
                'audio_generated': success,
                'audio_output_path': output_audio_path if success else None
            })
            
            print(f"Question {i+1} processed successfully!")
    
    # Save comprehensive results
    os.makedirs('../results', exist_ok=True)
    with open('../results/qa_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n === Processing Complete ===")
    print("Results saved to ../results/qa_results.json")
    
    # Print summary
    print("\n SUMMARY:")
    for lang_code in ['en', 'mr', 'fr']:
        audio_count = sum(1 for qa in results[lang_code]['questions_answers'] 
                         if qa['question_source'] == 'audio')
        text_count = len(results[lang_code]['questions_answers']) - audio_count
        print(f"{lang_code.upper()}: {audio_count} audio inputs, {text_count} text inputs")

if __name__ == "__main__":
    main()
