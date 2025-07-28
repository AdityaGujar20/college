from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import torch

class MultilingualQASystem:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.load_models()
    
    def load_models(self):
        """Load all three language models"""
        
        # English - FLAN-T5
        print("Loading FLAN-T5 for English...")
        self.tokenizers['en'] = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.models['en'] = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        # Marathi - MahaBERT (using multilingual BERT for QA)
        print("Loading MahaBERT for Marathi...")
        self.tokenizers['mr'] = AutoTokenizer.from_pretrained("l3cube-pune/marathi-bert")
        self.models['mr'] = AutoModelForQuestionAnswering.from_pretrained("l3cube-pune/marathi-bert")
        
        # French - CamemBERT
        print("Loading CamemBERT for French...")
        self.tokenizers['fr'] = AutoTokenizer.from_pretrained("camembert-base")
        self.models['fr'] = AutoModelForQuestionAnswering.from_pretrained("camembert-base")
    
    def answer_question(self, question, context, language):
        """Generate answer for given question and context"""
        
        if language == 'en':
            return self._answer_with_flan_t5(question, context)
        else:
            return self._answer_with_bert(question, context, language)
    
    def _answer_with_flan_t5(self, question, context):
        """Use FLAN-T5 for English QA - Fixed temperature warning"""
        prompt = f"Answer the following question based on the context:\nQuestion: {question}\nContext: {context}\nAnswer:"
        
        inputs = self.tokenizers['en'](
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.models['en'].generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        answer = self.tokenizers['en'].decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer (remove the prompt)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def _answer_with_bert(self, question, context, language):
        """Use BERT-based models for Marathi and French QA - Fixed offset_mapping error"""
        
        # Prepare inputs with better handling
        inputs = self.tokenizers[language](
            question, 
            context, 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
            add_special_tokens=True
        )
        
        if 'offset_mapping' in inputs:
            del inputs['offset_mapping']
        
        with torch.no_grad():
            outputs = self.models[language](**inputs)
        
        # Get answer span with improved logic
        start_scores = outputs.start_logits[0]
        end_scores = outputs.end_logits[0]
        
        # Find the best start and end positions
        start_idx = torch.argmax(start_scores).item()
        end_idx = torch.argmax(end_scores).item()
        
        # Ensure end comes after start
        if end_idx < start_idx:
            end_idx = start_idx + 10  # Default span of 10 tokens
        
        # Limit answer length to reasonable size
        if end_idx - start_idx > 50:
            end_idx = start_idx + 50
        
        # Ensure we don't go beyond input length
        max_len = len(inputs['input_ids'][0])
        end_idx = min(end_idx, max_len - 1)
        
        # Decode answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizers[language].decode(answer_tokens, skip_special_tokens=True)
        
        # Clean up the answer
        answer = answer.strip()
        
        if language == 'mr':
            question_words = ['काय', 'कसे', 'कोठे', 'केव्हा', 'कोण', 'किती']
            for word in question_words:
                if answer.startswith(word):
                    answer = answer[len(word):].strip()
        elif language == 'fr':
            # Remove French question words that might leak into answer
            question_words = ['Qu\'est-ce que', 'Comment', 'Où', 'Quand', 'Qui', 'Combien']
            for word in question_words:
                if answer.startswith(word):
                    answer = answer[len(word):].strip()
        
        if not answer or len(answer) < 3:
            return self._get_fallback_answer(context, language)
        
        if any(char in answer for char in ['##', '[UNK]', '[CLS]', '[SEP]']):
            return self._get_fallback_answer(context, language)
        
        return answer

    
    def _get_fallback_answer(self, context, language):
        """Generate fallback answer when main method fails"""
        try:
            # Get first meaningful sentence from context
            if language == 'mr':
                sentences = context.split('।')  # Marathi sentence separator
                for sentence in sentences[:3]:  # Check first 3 sentences
                    if len(sentence.strip()) > 20:
                        return sentence.strip()[:200] + "..."
            elif language == 'fr':
                sentences = context.split('.')
                for sentence in sentences[:3]:  # Check first 3 sentences
                    if len(sentence.strip()) > 20:
                        return sentence.strip()[:200] + "..."
            
            return context[:200].strip() + "..."
            
        except:
            return f"Unable to find answer in the context. (Language: {language})"
