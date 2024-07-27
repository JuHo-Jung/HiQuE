import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained("roberta-base")

    print("roberta Model Loaded.")


    data_list = sorted(glob("./data/all/*"), key=lambda x: int(x.split("/")[-1]))
    for i in tqdm(data_list):
        id = i.split("/")[-1]
        print(id)

        try:

            df = pd.read_csv(f"/media/dsail/dataset/daic-woz/{id}_P/{id}_TRANSCRIPT_FINAL.csv")
            q_list = sorted(glob(f"./data/all/{id}/*.csv"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
            for file in q_list:
                
                q_id = file.split("/")[-1].split(".")[0]
                q_df = df[df["topic_idx"] == int(q_id)]
                answer = q_df["answer"].values[0]
                encoded_input = tokenizer(answer, return_tensors='tf')
                cls_token = model(encoded_input)[1].numpy().ravel()
                np.save(f"./data/all/{id}/{q_id}_bert.npy", cls_token)
            

        except:
            np.save(f"./data/all/{id}/{q_id}_bert.npy", np.zeros(768))


           

if __name__ == '__main__':
    
    main()
    