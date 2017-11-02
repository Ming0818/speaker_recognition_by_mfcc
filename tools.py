import numpy as np
import librosa

def extract_mel_spectrogram(audio_file_path):
    try:
        x, fs = librosa.load(audio_file_path, sr=44100)
        mfccs = librosa.feature.mfcc(x, sr=fs)

        return mfccs
    except:
        print("Cannnot generate mel spectrogram from {}".format(audio_file_path))
        return None

def zero_pad(vector, maxlen):
    """
    this function assumption column is 20. Only check raw!
    """
    vector_shape = vector.shape

    if vector_shape[1] > maxlen:
        vector = vector[:, 0:maxlen]
        return vector

    pad_vector = np.zeros([vector_shape[0], maxlen - vector_shape[1]]).astype(np.float32)

    vector = np.hstack([vector, pad_vector])

    return vector
