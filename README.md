# HiQuE: Hierarchical Question Embedding Network for Multimodal Depression Detection


## Introduction 

## Code Description

- whisper_segment.py: extract transcript, timestamp (start and end point) from interview wav file
- preprocess.py: map
- split_wav.py: split interview sequence into each question-answer pairs
- similarity.py: bert-score based similarity question mapping (topic, topic idx, topic question) 
- Feature extraction: extract audio,
  - extract_audio.py: extract audio feature
  - extract_visual.py: extract visual feature
  - extract_bert.py: extract text feature
- fusionmodel.py: proposed model
- hique.py: training and testing the model
  

## Method
We initially utilized Whisper to transcribe WAV audio files. Subsequently, we assumed ellie as the questioner and divided the roles into the interviewer (ellie) and interviewee (participant). Although not executed due to time constraints, applying ASR algorithms during this process could potentially yield more accurate tagging. Next, we calculated the semantic similarity between the interviewer's questions and the 85 predefined questions from Appendix E using the bert-score metric. An example of the outcomes of this question mapping is as follows:


Before mapping (Raw) -> After mapping (Question Embedding)

ex1. You travel? -> Do you travel a lot (main) 

ex2.  Can you be a little bit more specific? ->  Can you give me an example of that (follow-up) 

ex3. What do you decide to do now? -> What do you do now (main) 

ex4. Where are you from? ->  Where are you from originally (main) 

ex5. How many of your kids? ->  Tell me about your kids (main) 


After completing the question mapping, the subsequent steps of the process (question elaboration, hierarchical position elaboration, feature extraction) remained consistent with the methodologies detailed in our paper. We then proceeded with binary classification to differentiate between depression and normal states.


## Limitations : 
There are several limitations to our extensive experiment. Firstly, since the text was transcribed based on audio, the accuracy of the text content may be compromised. Additionally, relying solely on audio makes it challenging to identify whether the interviewer or interviewee speaks precisely. Therefore, such errors could potentially introduce confusion in the model's training and validation processes.

## Findings : 
Based on this extensive experiment, our research validates the generalizability of our model in capturing depression cues from audio, even for questions not predefined.



# Citation 
