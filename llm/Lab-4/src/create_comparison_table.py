import json
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu

def load_results():
    """Load the QA results from JSON"""
    try:
        with open('../results/qa_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: qa_results.json not found. Run main.py first!")
        return None

def calculate_metrics(results):
    """Calculate ROUGE and BLEU scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    # Dummy reference answers for demonstration, replace with your real references
    reference_answers = {
        'en': [
            "Artificial intelligence has revolutionized healthcare by enabling faster diagnosis, personalized treatment plans, and improved patient outcomes.",
            "AI-powered systems can detect cancer cells in X-rays and MRI scans with 95% accuracy.",
            "Telemedicine and remote monitoring allow for continuous patient vital signs tracking and enable immediate intervention."
        ],
        'mr': [
            "कृत्रिम बुद्धिमत्ता आरोग्यसेवेत जलद निदान, वैयक्तिक उपचार योजना आणि सुधारित रुग्ण परिणाम सक्षम करते.",
            "AI प्रणाली एक्स-रे आणि MRI स्कॅनमध्ये 95% अचूकतेने कर्करोग पेशी शोधू शकतात.",
            "दूरवैद्यक आणि दूरस्थ निरीक्षण रुग्णांचे जीवनसत्त्वे सतत तपासतात आणि त्वरित हस्तक्षेप शक्य करतात."
        ],
        'fr': [
            "L'intelligence artificielle a révolutionné les soins de santé en permettant des diagnostics plus rapides, des plans de traitement personnalisés et de meilleurs résultats pour les patients.",
            "Les systèmes d'IA peuvent détecter les cellules cancéreuses dans les radiographies et les IRM avec une précision de 95%.",
            "La télémédecine et la surveillance à distance permettent un suivi continu des signes vitaux des patients et une intervention immédiate."
        ]
    }

    metrics = {}
    
    for lang in ['en', 'mr', 'fr']:
        qas = results.get(lang, {}).get('questions_answers', [])
        total_questions = len(qas)
        if total_questions == 0:
            print(f"No QA pairs found for {lang}")
            continue

        # Calculate success rates
        audio_input_success = sum(1 for qa in qas if qa.get('question_source') == 'audio') / total_questions * 100
        audio_output_success = sum(1 for qa in qas if qa.get('audio_generated', False)) / total_questions * 100

        # Average answer length
        avg_answer_length = sum(len(qa.get('answer', '')) for qa in qas) / total_questions

        # Quality score based on answer length and fallback phrases
        quality_count = 0
        for qa in qas:
            ans = qa.get('answer', "")
            if isinstance(ans, str) and len(ans) > 10 and not ans.lower().startswith("unable to find"):
                quality_count += 1
        answer_quality = (quality_count / total_questions) * 5  # scale 1 to 5

        # Compute ROUGE-1 and BLEU scores
        rouge_scores = []
        bleu_scores = []

        for i, qa in enumerate(qas):
            reference = reference_answers[lang][i] if i < len(reference_answers[lang]) else ""
            hypothesis = qa.get('answer', "")
            if not isinstance(hypothesis, str):
                hypothesis = str(hypothesis)
            hypothesis = hypothesis.strip()

            if hypothesis == "":
                rouge1_f = 0.0
                bleu_score = 0.0
            else:
                rouge_result = scorer.score(reference, hypothesis)
                rouge1_f = rouge_result['rouge1'].fmeasure
                try:
                    bleu_score = sentence_bleu([reference.split()], hypothesis.split()).score
                except Exception as e:
                    print(f"BLEU calculation error for lang={lang} question={i+1}: {e}")
                    bleu_score = 0.0

            rouge_scores.append(rouge1_f)
            bleu_scores.append(bleu_score)

        rouge_avg = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        bleu_avg = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        metrics[lang] = {
            'audio_input_success': round(audio_input_success, 1),
            'audio_output_success': round(audio_output_success, 1),
            'avg_answer_length': round(avg_answer_length, 1),
            'answer_quality': round(answer_quality, 1),
            'rouge1_avg': round(rouge_avg, 3),
            'bleu_avg': round(bleu_avg, 2)
        }
    
    return metrics

def create_comparison_table():
    """Create comprehensive comparison table"""
    
    print("Creating comparison table...")
    
    results = load_results()
    if not results:
        return
    
    metrics = calculate_metrics(results)
    
    comparison_data = {
        'Model': ['FLAN-T5', 'MahaBERT', 'CamemBERT'],
        'Language': ['English', 'Marathi', 'French'],
        'Model Type': ['Seq2Seq (T5-based)', 'BERT-based QA', 'RoBERTa-based QA'],
        'Model Size': ['780M parameters', '110M parameters', '110M parameters'],
        'Training Data': ['Mixed multilingual tasks', '752M tokens Marathi', '138GB French text'],
        'Audio Input Success (%)': [
            metrics.get('en', {}).get('audio_input_success', 0),
            metrics.get('mr', {}).get('audio_input_success', 0),
            metrics.get('fr', {}).get('audio_input_success', 0)
        ],
        'Audio Output Success (%)': [
            metrics.get('en', {}).get('audio_output_success', 0),
            metrics.get('mr', {}).get('audio_output_success', 0),
            metrics.get('fr', {}).get('audio_output_success', 0)
        ],
        'Avg Answer Length': [
            metrics.get('en', {}).get('avg_answer_length', 0),
            metrics.get('mr', {}).get('avg_answer_length', 0),
            metrics.get('fr', {}).get('avg_answer_length', 0)
        ],
        'Answer Quality (1-5)': [
            metrics.get('en', {}).get('answer_quality', 0),
            metrics.get('mr', {}).get('answer_quality', 0),
            metrics.get('fr', {}).get('answer_quality', 0)
        ],
        'ROUGE-1 Score': [
            metrics.get('en', {}).get('rouge1_avg', 0),
            metrics.get('mr', {}).get('rouge1_avg', 0),
            metrics.get('fr', {}).get('rouge1_avg', 0)
        ],
        'BLEU Score': [
            metrics.get('en', {}).get('bleu_avg', 0),
            metrics.get('mr', {}).get('bleu_avg', 0),
            metrics.get('fr', {}).get('bleu_avg', 0)
        ],
        'Coherence (Manual 1-5)': [5, 3, 4],
        'Relevance (Manual 1-5)': [5, 3, 4],
        'Fluency (Manual 1-5)': [5, 3, 4],
        'Voice Clarity (Manual 1-5)': [5, 3, 4],
        'Pronunciation (Manual 1-5)': [5, 2, 4],
        'Response Time': ['Fast', 'Moderate', 'Fast'],
        'Noted Issues': [
            'Temperature warnings (fixed)',
            'QA layer not fine-tuned, Devanagari issues',
            'Incomplete responses, TTS connectivity issues'
        ]
    }

    df = pd.DataFrame(comparison_data)
    
    df.to_csv('../results/comparison_table.csv', index=False, encoding='utf-8')
    
    print("\nMULTILINGUAL VOICE-BASED QA SYSTEM - COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    
    analysis = {
        'summary': {
            'total_languages': 3,
            'total_questions_processed': sum(len(results.get(lang, {}).get('questions_answers', [])) for lang in ['en', 'mr', 'fr']),
            'overall_audio_input_success': sum(metrics.get(lang, {}).get('audio_input_success', 0) for lang in metrics)/3,
            'overall_audio_output_success': sum(metrics.get(lang, {}).get('audio_output_success', 0) for lang in metrics)/3,
            'average_rouge1_score': sum(metrics.get(lang, {}).get('rouge1_avg', 0) for lang in metrics)/3,
            'average_bleu_score': sum(metrics.get(lang, {}).get('bleu_avg', 0) for lang in metrics)/3,
        },
        'metrics': metrics
    }

    with open('../results/detailed_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"\nComparison table saved to: ../results/comparison_table.csv")
    print(f"Detailed analysis saved to: ../results/detailed_analysis.json")
    
    return df, analysis

def print_summary_stats(results):
    """Print summary statistics"""
    print("\nSUMMARY STATISTICS")
    print("-" * 40)
    
    for lang in ['en', 'mr', 'fr']:
        lang_name = {'en': 'English', 'mr': 'Marathi', 'fr': 'French'}[lang]
        questions = results.get(lang, {}).get('questions_answers', [])
        
        print(f"\n{lang_name}:")
        print(f"  Questions processed: {len(questions)}")
        
        audio_inputs = sum(1 for qa in questions if qa.get('question_source') == 'audio')
        text_inputs = len(questions) - audio_inputs
        print(f"  Audio inputs: {audio_inputs}, Text inputs: {text_inputs}")
        
        successful_audio_outputs = sum(1 for qa in questions if qa.get('audio_generated', False))
        print(f"  Successful audio outputs: {successful_audio_outputs}/{len(questions)}")
        
        avg_answer_len = sum(len(qa.get('answer', '')) for qa in questions) / len(questions) if questions else 0
        print(f"  Average answer length: {avg_answer_len:.1f} characters")

if __name__ == "__main__":
    results = load_results()
    if results:
        print_summary_stats(results)
        create_comparison_table()
