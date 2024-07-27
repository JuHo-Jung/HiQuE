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

1. Transcription: Used Whisper to transcribe WAV audio files.

2. Role Identification: Assumed 'ellie' as the questioner, distinguishing interviewer (ellie) and interviewee (participant).

- Note: Implementing ASR algorithms could improve tagging accuracy.

3. Semantic Similarity: Calculated similarity between interviewer's questions and 85 predefined questions from Appendix E using the BERT-score metric.

- Example Mappings:
  - Before: "You travel?" -> After: "Do you travel a lot (main)"
  - Before: "Can you be a little bit more specific?" -> After: "Can you give me an example of that (follow-up)"
  - Before: "What do you decide to do now?" -> After: "What do you do now (main)"
  - Before: "Where are you from?" -> After: "Where are you from originally (main)"
  - Before: "How many of your kids?" -> After: "Tell me about your kids (main)"
    
4. Subsequent Steps: Followed methodologies from our paper for question elaboration, hierarchical position elaboration, and feature extraction.

5. Classification: Conducted binary classification to distinguish between depression and normal states.


## Limitations : 
There are several limitations to our extensive experiment. Firstly, since the text was transcribed based on audio, the accuracy of the text content may be compromised. Additionally, relying solely on audio makes it challenging to identify whether the interviewer or interviewee speaks precisely. Therefore, such errors could potentially introduce confusion in the model's training and validation processes.

## Findings : 
Based on this extensive experiment, our research validates the generalizability of our model in capturing depression cues from audio, even for questions not predefined.



# Citation 
