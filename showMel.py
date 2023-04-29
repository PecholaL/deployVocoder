import librosa
import torch
import numpy
import pickle
import matplotlib.pyplot

sr = 22050
hop_size = 256
ref_db  = 20
max_db  = 100

f = open('mel.pkl', 'rb')
mel = pickle.load(f)
f.close()

mel = abs(mel.squeeze())
mel = numpy.clip((mel - ref_db + max_db) / max_db, 1e-8, 10)

print(mel.shape)

mel_db = librosa.power_to_db(mel, ref=numpy.max)
librosa.display.specshow(mel_db, sr=sr, hop_length=hop_size, x_axis='time', y_axis='mel')
matplotlib.pyplot.colorbar()
matplotlib.pyplot.show()