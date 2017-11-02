import sys
import glob
import numpy as np
import os
import network
import pickle
from tools import extract_mel_spectrogram, zero_pad

def pad_sequences(vector_list, maxlen):
    pad_vector_list = []
    for vector in vector_list:
        pad_vector = zero_pad(vector, maxlen=700)
        pad_vector_list.append(pad_vector)
    
    return pad_vector_list

if __name__ == '__main__':
    audio_file_dir = sys.argv[1]
    mel_spectrogram_list = []
    # stupid, need fix
    # hoge, foo, bar, fuga is dummy
    hoge = [1, 0, 0, 0]
    foo = [0, 1, 0, 0]
    bar = [0, 0, 1, 0]
    fuga = [0, 0, 0, 1]
    y = []

    if os.path.exists("x.pkl") and os.path.join("y.pkl"):
        print("Found pre-generated data, using these pkl files.")
        with open("./x.pkl", "rb") as x_pkl_file:
            x = pickle.load(x_pkl_file)
        with open("./y.pkl", "rb") as y_pkl_file:
            y = pickle.load(y_pkl_file)
    else:
        for audio_file_path in glob.glob(os.path.join(audio_file_dir, "*", "*.m4a")):
            if "hoge" in audio_file_path:
                y.append(hoge)
            elif "foo" in audio_file_path:
                y.append(foo)
            elif "bar" in audio_file_path:
                y.append(bar)
            elif "fuga" in audio_file_path:
                y.append(fuga)
            else:
                print("{} is unknown tag, skip.".format(audio_file_path))
                continue
            mel_spectrogram = extract_mel_spectrogram(audio_file_path)
            mel_spectrogram /= mel_spectrogram.max()
            mel_spectrogram_list.append(mel_spectrogram)

            print("Processing => {}(shape={})".format(audio_file_path, mel_spectrogram.shape))

        x = pad_sequences(mel_spectrogram_list, maxlen=700)
        x = np.array(x)
        y = np.array(y)

        with open("./x.pkl", "wb") as x_pkl_file:
            pickle.dump(x, x_pkl_file)

        with open("./y.pkl", "wb") as y_pkl_file:
            pickle.dump(y, y_pkl_file)

    # reshape(need?)
    x = x.reshape(-1, 20, 700, 1)

    print("x.shape={}".format(x.shape))
    print("y.shape={}".format(y.shape))
    network.train(x, y)
