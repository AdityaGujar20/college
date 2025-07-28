# Multilingual Voice-Based Document Question Answering System

## Overview

This project implements a **comparative study** of transformer-based language models for voice-based Question Answering (QA) on PDF documents across three languages: **English**, **Marathi (Indic)**, and **French (International Non-English)**.

The system accepts PDF documents as input, processes spoken or text questions in the respective language via speech-to-text or direct text input, extracts answers using transformer models, and returns voice output via text-to-speech synthesis. The project highlights differences in language model architecture, language characteristics, and voice processing capabilities.

## Project Objectives

- **Model Evaluation:** Compare three transformer-based language models:
  - **English Foundation Model:** FLAN-T5
  - **Indic Language Model (Marathi):** MahaBERT
  - **International Model (French):** CamemBERT

- **Multimodal Inputs:** Take user questions as either voice input (audio file) or direct text input.

- **Multimodal Outputs:** Return answers as both text and synthesized speech.

- **Document Processing:** Parse text content from PDF documents in all three languages.

- **Cross-language Comparability:** Use consistent question sets across languages and measure performance quantitatively and qualitatively.

## System Components & Technologies

| Component             | Technology/Tool                           |
|-----------------------|-----------------------------------------|
| Transformer Models     | FLAN-T5 (English), MahaBERT (Marathi), CamemBERT (French) |
| Speech-to-Text (STT)  | OpenAI Whisper (offline)                 |
| Text-to-Speech (TTS)  | gTTS (free, online) with pyttsx3 fallback (offline) |
| PDF Text Extraction   | pdfplumber                              |
| Evaluation Metrics    | ROUGE (rouge-score), BLEU (sacrebleu)  |
| Programming Language  | Python 3                               |
| Libraries             | transformers, torch, pydub, nltk, pandas |

## Project Flow

### 1. Data Preparation

- Three PDFs created on the same topic ("Artificial Intelligence in Modern Healthcare") in English, Marathi, and French.
- Synthetic documents ensure consistent domain knowledge across languages.

### 2. Question Input

- Questions are available in both audio (MP3) and text formats.
- Sample questions crafted for each language, ensuring semantic equivalence.

### 3. Speech-to-Text (STT)

- Uses **OpenAI Whisper (offline)** for robust transcription of voice questions to text.
- Supports all three languages without online API dependencies.

### 4. Document Text Extraction

- Extract plaintext from PDF files using `pdfplumber`.
- Text normalized and truncated to fit model input size constraints.

### 5. Question Answering (QA)

- **English:** Uses FLAN-T5 sequence-to-sequence model.
- **Marathi:** Uses MahaBERT fine-tuned for question answering.
- **French:** Uses CamemBERT adapted for question answering.
- Models extract relevant answers from context for input questions.

### 6. Text-to-Speech (TTS)

- Answers synthesized into speech using **gTTS** (Google Text-to-Speech free tier) with **pyttsx3** offline fallback.
- Supports language-specific pronunciation and voice characteristics.

### 7. Output & Evaluation

- Text and audio answers saved for analysis.
- Quantitative evaluation metrics include:
  - **ROUGE-1** scores for content overlap
  - **BLEU** scores for n-gram precision
- Human evaluation metrics collected for coherence, relevance, fluency, voice clarity, and pronunciation.

## Folder Structure

```
voice-qa-system/
├── data/
│   ├── pdfs/
│   │   ├── english.pdf
│   │   ├── marathi.pdf
│   │   └── french.pdf
│   └── audio/
│       ├── input/       # Voice questions (MP3 files)
│       └── output/      # Generated voice answers
├── models/
│   ├── english/
│   ├── marathi/
│   └── french/
├── results/
│   ├── qa_results.json       # Collected QA outputs
│   ├── comparison_table.csv  # Metric comparison table
│   └── detailed_analysis.json
├── src/
│   ├── pdf_processing.py
│   ├── speech_processing.py
│   ├── qa_models.py
│   ├── sample_questions.py
│   ├── create_audio_questions.py
│   ├── create_comparison_table.py
│   └── main.py
├── requirements.txt
└── README.md
```

## How to Run

### Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Step 1: Create Audio Questions (optional but recommended)

```bash
python src/create_audio_questions.py
```

This generates audio files for your sample questions.

### Step 2: Run the Main QA Pipeline

```bash
python src/main.py
```

This runs the full system:
- Extracts PDF text
- Accepts questions (audio or text)
- Generates answers
- Synthesizes voice answers
- Saves results and audio files

### Step 3: Generate Comparison & Evaluation Table

```bash
python src/create_comparison_table.py
```

This computes ROUGE and BLEU scores, builds a results table, and saves the analysis.

## Evaluation Metrics

- **ROUGE-1:** Measure of word-level overlap
- **BLEU:** N-gram precision metric for generated text
- **Human Evaluation:** Coherence, relevance, fluency, voice clarity, pronunciation rated on scale 1-5

## Conclusion

This project demonstrates a fully functional multilingual voice-based document QA system that:

- Integrates advanced transformer models for diverse languages
- Supports flexible input modalities (text and voice)
- Employs offline speech-to-text and text-to-speech for accessibility
- Provides quantitative and qualitative evaluations for a comparative study

It showcases the challenges and solutions in building multilingual NLP pipelines with voice interfaces.

## Contact / Support

For questions or issues, please contact [Your Name] at [Your Email].

Feel free to customize or expand this `README.md` based on your presentation style or additional insights!

Sources
