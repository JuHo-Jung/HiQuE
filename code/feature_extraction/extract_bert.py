import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained("bert-base-uncased")

    print("BERT Model Loaded.")


    data_list = sorted(glob("./data/embed_text/*"), key=lambda x: int(x.split("/")[-1].split("_")[0]))
    for i in tqdm(data_list):
        id = i.split("/")[-1].split("_")[0]
        print(id)
        os.makedirs(f"./feature_extraction/text/{id}", exist_ok=True)
        try:

            df = pd.read_csv(f"./data/embed_text/{id}_P.csv")

            for i,row in df.iterrows():
                
                q_id = int(row["topic_idx"])
                answer = df.iloc[i+1]["text"]
                encoded_input = tokenizer(answer, return_tensors='tf')
                cls_token = model(encoded_input)[1].numpy().ravel()
                np.save(f"./feature_extraction/text/{id}/{q_id}_bert.npy", cls_token)
            

        except:
            np.save(f"./feature_extraction/text/{id}/{q_id}_bert.npy", np.zeros(768))


           

if __name__ == '__main__':
    
    main()
    