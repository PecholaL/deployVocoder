import torch
import json
import pickle
from scipy.io.wavfile import write
from hifigan.models import Generator
from hifigan.env import AttrDict

MAX_WAV_VALUE = 32768.0

device = 'cpu'

with open('config.json') as f:
	config = f.read()
json_config = json.loads(config)
h = AttrDict(json_config)

generator = Generator(h).to(device)
state_dict_g = torch.load('generator_v3', device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()

fin = open('mel.pkl', 'rb')
mel = pickle.load(fin)
print(mel.shape)
fin.close()
with torch.no_grad():
	y_g_hat = generator(mel)
	audio = y_g_hat.squeeze()
	audio = audio * MAX_WAV_VALUE
	audio = audio.cpu().numpy().astype('int16')
	write('recon.wav', h.sampling_rate, audio)
