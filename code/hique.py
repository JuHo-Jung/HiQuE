import os
# gpu setting

import argparse
from glob import glob
import random
from PIL import Image
import numpy as np
from tensorflow.python.keras.layers.merge import multiply
from tqdm import tqdm
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from fusionmodel import  hique
from numpy.random import seed
import matplotlib.pyplot as plt



use_gpu = "1"
learning_rate = 2e-4
epochs = 50
batch_size = 2
MAX_SEQ_LEN = 85
SEED = 9821
# SEED = 3401


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu


def drop_zeros(X, size):
    ids = []
    for idx, row in enumerate(X):
        if not np.array_equal(row, np.zeros([size])):
            ids.append(idx)
    return X[ids]


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def get_features(daic):
    
    daic = daic[['Participant_ID', 'PHQ_Binary', 'PHQ_Score', 'Gender']]
    audio_features = []
    visual_features = []
    text_features = []
    y_deprssion = []
    y_score = []
    y_gender = []

    for row in daic.values:

        name, binary, score, gender = row

        audio_features = []
        for i in range(85):
            if os.path.exists(f"./feature_extraction/audio/{name}/{i}.csv"):
                df = pd.read_csv(f"./feature_extraction/audio/{name}/{i}.csv")
                audio_features.append(df.iloc[0].values)
            else:
                audio_features.append(np.zeros(88))

        text_features = []
        for i in range(85):
            if os.path.exists(f"./feature_extraction/text/{name}/{i}_bert.npy"):
                text_features.append(np.load(f"./feature_extraction/text/{name}/{i}_bert.npy"))
            else:
                text_features.append(np.zeros(768))


        visual_features = []
        for i in range(85):
            if os.path.exists(f"./feature_extraction/video/{name}/{i}_v.npy"):
                visual_features.append(np.load(f"./feature_extraction/video/{name}/{i}_v.npy"))
            else:
                visual_features.append(np.zeros(4096))

        # y_deprssion.append(binary)
        if binary == 0:
            y_deprssion.append([1, 0])
        else:
            y_deprssion.append([0, 1])


        y_score.append(score)
        if gender == 0:
            y_gender.append([1, 0])
        else:
            y_gender.append([0, 1])

    return np.array(audio_features), np.array(visual_features), np.array(text_features), np.array(y_deprssion), np.array(y_score), np.array(y_gender)


def class_report(y_pred,y_true,name):
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

    sensitivty = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    with open(f'./result/classification_reports/{name}_cr.txt', 'w') as f:
        print(classification_report(y_true, y_pred), file=f)
        print("", file=f)
        print("sensitivity: ", sensitivty, file=f)
        print("specificity: ", specificity, file=f)
        print("precision: ", precision, file=f)
        print("f1: ", f1, file=f)
        print("macro_precision: ", macro_precision, file=f)
        print("macro_recall: ", macro_recall, file=f)
        print("macro_f1: ", macro_f1, file=f)
        print("", file=f)
        print("Confusion Matrix: ",file=f)
        print(confusion_matrix(y_true, y_pred), file=f)



def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in np.array(points):
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_history(train_loss, val_loss, name, save_dir):
    plt.figure()
    x_len = np.arange(len(val_loss))
    plt.plot(x_len, smooth_curve(val_loss), marker='.', c='red', label=f"val_{name}")
    plt.plot(x_len, smooth_curve(train_loss), marker='.', c='blue', label=f"train_{name}")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.savefig(f"./result/figure/{save_dir}_loss.png")


def draw_confusion_matrix(y_true, y_pred, path, title, classname):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    df_cm = pd.DataFrame(cm, index=classname, columns=classname)
    plt.figure(figsize=(5.5, 4))
    sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)




def main(options):

    epochs = options.epochs
    batch_size = options.batch_size
    learning_rate = options.lr
    name = options.name
    loss = options.loss
    dropout = options.dropout
    SEED = options.seed
    
    name = name + "_" + str(epochs) + "_" + str(dropout) +"_"+ str(learning_rate) +"_"+ str(batch_size) +"_"+ str(SEED)


    train_df = pd.read_csv("./labels/train_split.csv")
    val_df = pd.read_csv("./labels/dev_split.csv")
    test_df = pd.read_csv("./labels/test_split.csv")


    X_train_audio, X_train_visual, X_train_text, y_depression_train, y_score_train, y_gender_train = get_features(train_df)
    X_test_audio, X_test_visual, X_test_text, y_depression_test, y_score_test, y_gender_test = get_features(test_df)
    X_val_audio, X_val_visual, X_val_text, y_depression_val, y_score_val, y_gender_val = get_features(val_df)


    print("Train dataset")
    print(X_train_audio.shape, X_train_visual.shape, X_train_text.shape, y_depression_train.shape, y_score_train.shape, y_gender_train.shape)
    print("")

    print("Test dataset")
    print(X_test_audio.shape, X_test_visual.shape, X_test_text.shape, y_depression_test.shape, y_score_test.shape, y_gender_test.shape)
    print("")

    print("Validation dataset")
    print(X_val_audio.shape, X_val_visual.shape, X_val_text.shape, y_depression_val.shape, y_score_val.shape, y_gender_val.shape)
    print("")


    model = hique(MAX_SEQ_LEN, learning_rate, loss, dropout)


    checkpoint_filepath = f"./model/{name}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True
    )

    history = model.fit(
        [X_train_audio, X_train_visual, X_train_text], y_depression_train, batch_size=batch_size, shuffle=True,
        steps_per_epoch=len(X_train_audio) // batch_size,
        epochs=epochs,
        validation_data=([X_val_audio, X_val_visual,X_val_text], y_depression_val),
        callbacks=[checkpoint_callback]
    )

    model.load_weights(checkpoint_filepath)

    y_depression_pred= model.predict([X_test_audio, X_test_visual, X_test_text])

    y_depression_pred_for_save = pd.DataFrame(y_depression_pred)
    y_depression_pred_for_save.to_csv(f"./result/softmax/y_pred.csv", index=False)

    Y_pred = np.argmax(y_depression_pred, axis=1)

    Y_pred_for_save = pd.DataFrame(Y_pred)
    Y_pred_for_save.to_csv(f"./result/softmax/y_pred_argmax.csv", index=False)
    
    Y_test = np.argmax(y_depression_test, axis=1)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

    draw_confusion_matrix(y_true=Y_test, y_pred=Y_pred, path=f"./result/figure/cm/{name}_confusion_matrix.png", title="Audio+Vision+Text", classname=["Normal","Depression"])

    #classificatio_report
    class_report(Y_test, Y_pred, name)
    

    #graph
    plot_history(history.history['loss'], history.history['val_loss'], 'loss',name+"loss")
    plot_history(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', name+"acc")


if __name__ == "__main__":
    
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument("--gpu", type=str, default="1")
    OPTIONS.add_argument("--lr", type=float, default=2e-4)
    OPTIONS.add_argument("--epochs", type=int, default=50)
    OPTIONS.add_argument("--batch_size", type=int, default=8)
    OPTIONS.add_argument("--max_seq_len", type=int, default=85)
    OPTIONS.add_argument("--name", type=str, default="av")
    OPTIONS.add_argument("--loss", type=str, default="binary_crossentropy")
    OPTIONS.add_argument("--dropout", type=float, default=0.5)
    OPTIONS.add_argument("--seed", type=int, default=42)

    SEED = OPTIONS.parse_args().seed

    set_global_determinism(SEED)

    main(OPTIONS.parse_args())



    
