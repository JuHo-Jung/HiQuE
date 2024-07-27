import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import os 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def return_standardized_vectors(data):
    scaler = StandardScaler()
    scaler.fit(np.array(data).reshape(-1, 1))
    scaled_arr = scaler.transform(np.array(data).reshape(-1, 1))
    return np.array(scaled_arr).ravel()



def visual_feature(id):

    os.makedirs(f"./feature_extraction/video/{id}", exist_ok=True)


    df = pd.read_csv(f"./data/embed_text/{id}_P.csv")
    v = pd.read_csv(f"./data/{id}_P/features/{id}_vgg16.csv")
    v.dropna(inplace=True)

    for i,row in df.iterrows():
            
        q_id = int(row["topic_idx"])
        start = float(row["start"])
        end = float(row["end"])


        v_df = v[(v["timeStamp"] >= 1) & (v["timeStamp"] <= 4)].iloc[:,2:].mean(axis=0).values
        v_df = return_standardized_vectors(v_df)
        
            
        np.save(f"./feature_extraction/video/{id}/{q_id}_v.npy", v_df)


if __name__ == "__main__":

    data_list = sorted(glob("./data/embed_text/*"), key=lambda x: int(x.split("/")[-1].split("_")[0]))
    for i in tqdm(data_list):
        id = i.split("/")[-1].split("_")[0]
        visual_feature(id)
        

