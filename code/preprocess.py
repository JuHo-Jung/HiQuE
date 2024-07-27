import pandas as pd 
from glob import glob
from tqdm import tqdm







data_df = sorted(glob("./whisper/data/text/*"), key=lambda x: int(x.split("/")[-1].split("_")[0]))



for p in tqdm(data_df):
    df = pd.read_csv(p)
    col = df.columns
    new_df = pd.DataFrame(columns=col)
    
    agg_text = []
    agg_speaker = []
    agg_start = []
    agg_end = []
    
    for i,row in df.iterrows():
        start = row["start"]
        end = row["end"]
        speaker = row["speaker"]
        text = row["text"]
        
        if i == 0:
            agg_text.append(text)
            agg_speaker.append(speaker)
            agg_start.append(start)
            agg_end.append(end)
        
        else: 
            if speaker == agg_speaker[-1]:
                agg_text[-1] += " " + text
                agg_end[-1] = end
            else:
                agg_text.append(text)
                agg_speaker.append(speaker)
                agg_start.append(start)
                agg_end.append(end)
    
    new_df["start"] = agg_start
    new_df["end"] = agg_end
    new_df["speaker"] = agg_speaker
    new_df["text"] = agg_text
    
    new_df.to_csv("./whisper/data/cleaned_text/"+p.split("/")[-1], index=False)
