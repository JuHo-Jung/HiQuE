import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
import os
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus= tf.config.experimental.list_physical_devices('GPU')


def load_glove(path):
    embed_dict = {}
    with open(path,'r',encoding='utf8') as f:
        for line in tqdm(f):
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:],'float32')
                embed_dict[word]=vector
            except:
                f.__next__()
    return embed_dict


def get_w2v(sentence, model):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words    in input sentence.
    """
    return np.array([model.get(val, np.zeros(300)) for val in sentence.split()], dtype=np.float64)


def text_feature(id,glove):
    try:

        df = pd.read_csv(f"./dataset/daic-woz/{id}_P/{id}_TRANSCRIPT_FINAL.csv")
        q_list = sorted(glob(f"./data/all/{id}/*.csv"), key=lambda x: int(x.split("/")[-1].split(".")[0]))

        for file in q_list:
            
            q_id = file.split("/")[-1].split(".")[0]
            q_df = df[df["topic_idx"] == int(q_id)]
            answer = q_df["answer"].values[0]
            textual_features = get_w2v(answer, glove)

            np.save(f"./data/all/{id}/{q_id}_t.npy", textual_features)
            
    except:
        pass







if __name__ == "__main__":

    print("Start loading Glove...")
    glove = load_glove("./MISA/glove.840B.300d.txt")
    print("Glove loaded.")
    
    data_list = sorted(glob("./data/all/*"), key=lambda x: int(x.split("/")[-1]))
    for i in tqdm(data_list):
        id = i.split("/")[-1]
        print(id)
        text_feature(id,glove)
    