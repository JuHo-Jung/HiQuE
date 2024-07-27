import whisper
from glob import glob 
import pandas as pd 
from tqdm import tqdm



model = whisper.load_model("base")

def split_sentence(text):
    text = text.replace('"', '')
    sentences = text.split('?')

    sentences = [s.replace('"', '') for s in sentences]
    return sentences



p_id = sorted(glob("./data/*"), key=lambda x: int(x.split("/")[-1].split("_")[0]))
for p in tqdm(p_id):
    id = p.split("/")[-1].split("_")[0]
    result = model.transcribe(p+f"/{id}_AUDIO.wav")
    
    col = ['start', 'end', 'text']
    df = pd.DataFrame(columns = col)

    for i in range(len(result["segments"])):
        df.loc[i] = [result["segments"][i]['start'], result["segments"][i]['end'], result["segments"][i]['text']]

    df.to_csv(f"./whisper/segment/{id}_P.csv",index=False)

    # post processing / if Question : Ellie alse Participant

    for i,row in df.iterrows():
        if row["text"].endswith("?"):
            df.loc[i, "speaker"] = "Ellie"

        else:
            df.loc[i, "speaker"] = "Participant"


    df.to_csv(f"./whisper/data/text/{id}_P.csv",index=False)

