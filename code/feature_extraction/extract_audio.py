import os
import time
from glob import glob
import numpy as np
import pandas as pd
import audb
import audiofile
import opensmile
from tqdm import tqdm


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

data = sorted(glob("./whisper/data/wav/*"))


for file in tqdm(data):

    file_data = sorted(glob(f"{file}/*.wav"))
    id = file.split('/')[-1].split('.')[0]

    if os.path.exists(f"./whisper/feature_extraction/audio/{id}"):
        pass
    else:
        os.mkdir(f"./whisper/feature_extraction/audio/{id}")
            
        for file_split in file_data:

            signal, sampling_rate = audiofile.read(
                file = file_split,
                always_2d = True,
            )

            data = smile.process_signal(
                signal = signal,
                sampling_rate = sampling_rate,
            )

            data.to_csv(f"./whisper/feature_extraction/audio/{id}/{file_split.split('/')[-1].split('.')[0]}.csv",index=False)
        
