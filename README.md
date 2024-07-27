# HiQuE: Hierarchical Question Embedding Network for Multimodal Depression Detection

Official code for [ACM CIKM 2024] HiQuE

## Introduction 

HiQuE is designed for multimodal depression detection through hierarchical question embedding.

## Code Description

### HiQuE Components

- whisper_segment.py: Extracts transcript and timestamps (start and end points) from interview WAV files.
- preprocess.py: Maps data.
- split_wav.py: Splits interview sequences into individual question-answer pairs.
- similarity.py: Performs BERT-score-based similarity question mapping (topic, topic index, topic question).
- Feature extraction: Extracts audio, visual, and text features:
  - extract_audio.py: Extracts audio features.
  - extract_visual.py: Extracts visual features.
  - extract_bert.py: Extracts text features.
- fusionmodel.py: Contains the proposed model.
- hique.py: Handles model training and testing.

### Embedding 

Code for text and audio embedding using various encoders.


## Method

1. Transcription: Whisper was used to transcribe WAV audio files.

2. Role Identification: Assumed 'Ellie' as the questioner, distinguishing interviewer (Ellie) and interviewee (participant).

  - Note: Implementing ASR algorithms could improve tagging accuracy.

3. Semantic Similarity: The similarity between the interviewer's questions and 85 predefined questions from Appendix E was calculated using the BERT-score metric.

  - Example Mappings:
    - Before: "You travel?" -> After: "Do you travel a lot (main)"
    - Before: "Can you be a little bit more specific?" -> After: "Can you give me an example of that (follow-up)"
    - Before: "What do you decide to do now?" -> After: "What do you do now (main)"
    - Before: "Where are you from?" -> After: "Where are you from originally (main)"
    - Before: "How many of your kids?" -> After: "Tell me about your kids (main)"
    
4. Subsequent Steps: Followed methodologies from our paper for question elaboration, hierarchical position elaboration, and feature extraction.

5. Classification: Conducted binary classification to distinguish between depression and normal states.


## Limitations : 
Several limitations were noted during our experiments:

- Transcription Accuracy: As text was transcribed from audio, inaccuracies may exist in the text content.
- Role Identification: Reliance on audio alone makes it difficult to accurately distinguish between the interviewer and interviewee, potentially causing confusion in model training and validation.

## Findings : 
Our research demonstrates that the HiQuE model can generalize well in detecting depression cues from audio, even for questions that were not predefined.


# Citation 
Please cite our work as follows:

