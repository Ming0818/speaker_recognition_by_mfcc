from keras.models import model_from_json
from keras.models import load_model
import sys
import numpy as np
import os
from tools import extract_mel_spectrogram, zero_pad
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# hide annoying verbose message

def predict(target_vector, model):
    label_list = ['hoge', 'foo', "bar", "fuga"]
    predict_result = model.predict(target_vector)
    predict_index = np.where(predict_result == predict_result.max())[1][0]
    print("Predict ===> {}".format(label_list[predict_index]))

if __name__ == '__main__':
    model_json_path = "./model.json"
    model_param_path = "./params.hdf5"

    with open(model_json_path, 'r') as model_json_file:
        model_json_str = model_json_file.read()

    model = model_from_json(model_json_str)
    model.load_weights(model_param_path)

    audio_file_path = sys.argv[1]
    mel_spectrogram = extract_mel_spectrogram(audio_file_path)
    mel_spectrogram /= mel_spectrogram.max()
    pad_mel_spectrogram = zero_pad(mel_spectrogram, maxlen=700)
    pad_mel_spectrogram = pad_mel_spectrogram.reshape(1, 20, 700, 1)

    predict(pad_mel_spectrogram, model)

