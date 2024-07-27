import numpy as np
import librosa
import soundfile as sf
import pandas as pd 
from glob import glob
import os 
from tqdm import tqdm


def open_wav(file_name, origin_rate):
    data, rate = librosa.load(file_name, sr=origin_rate)
    
    return data, rate



def split_wav(data, sample_rate, start, end):
    start = start*sample_rate
    end = end*sample_rate
    start = int(start)
    end = int(end)

    return data[start:end]


def get_info(data):
    #input: wav file 한개 
    id = data.split("/")[-1].split("_")[0]
    df = pd.read_csv(f"./whisper/data/cleaned_text/{id}_P.csv",encoding='cp949')

    wav_data, sample_rate = open_wav(data, 16000)

    if os.path.exists(f"./whisper/data/wav/{id}"):
        pass
    else:
        os.mkdir(f"./whisper/data/wav/{id}")
        
        df = df.dropna()
        for i in range(len(df)):
            start = df.iloc[i,0]
            end = df.iloc[i,1]
            wav = split_wav(wav_data, sample_rate, start, end)

            sf.write(f"./whisper/data/wav/{id}/{i}.wav", wav, 16000)
        




if __name__ == '__main__':

    total_id = sorted(glob("./whisper/data/cleaned_text/*"), key=lambda x: int(x.split("/")[-1].split("_")[0]))
    for id in tqdm(total_id):
        pid = id.split("/")[-1].split("_")[0]
        if os.path.exists(f"./whisper/data/cleaned_text/{pid}_P.csv"):
            get_info(f"./data/{pid}_P/{pid}_AUDIO.wav")
        else:
            print(f"{pid} does not have transcript")