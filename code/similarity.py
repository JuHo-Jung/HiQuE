import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
import os
from evaluate import load
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")

print("BERT Model Loaded.")


bertscore = load("bertscore")


df = pd.read_csv("./daic-woz/300_P/300_TRANSCRIPT_FINAL.csv")
df_col = ['topic_idx', 'topic',	'question_x', 'type']
df = df[df_col]

ques = df["question_x"].tolist()
q_type = df["type"].tolist()



cleaned_text = sorted(glob("./whisper/data/cleaned_text/*"), key=lambda x: int(x.split("/")[-1].split("_")[0]))


for text in tqdm(cleaned_text):

    pid = text.split("/")[-1].split("_")[0]
    tmp_df = pd.read_csv(text)
    copy_df = tmp_df.copy()
    

    for i, row in copy_df.iterrows():

        speaker = row["speaker"]
        if speaker == "Participant":
            copy_df.loc[i, "topic_idx"] = int(0)
            copy_df.loc[i, "topic"] = int(0)
        
        else:
            simliarity = []
            text = row["text"]

            for j, q in enumerate(ques):
                bert_score = bertscore.compute(predictions=[text], references=[q], lang="en")
                sim = bert_score["f1"][0]
                simliarity.append(sim)


            simliarity = np.array(simliarity).ravel()
            idx = np.argmax(simliarity)

            copy_df.loc[i, "topic_idx"] = int(idx)
            copy_df.loc[i, "topic"] = ques[idx]
            copy_df.loc[i, "type"] = q_type[idx]


    copy_df.to_csv("./whisper/data/embed_text/"+pid+"_P.csv", index=False)