import torch
import librosa
import pickle

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
    
def spectral_normalize_torch(x, C=1, clip_val=1e-5):
    output = torch.log(torch.clamp(x, min=clip_val) * C)
    return output
    

#----------------------------------------main
device = 'cpu'
sr = 22050
fmin = 0
fmax = 8000
n_fft = 1024
n_mels = 80
hop_size = 256
win_size = 1024

wav, sr = librosa.load('test.wav', sr=sr)
wav = torch.FloatTensor(wav).to(device)
x = mel_spectrogram(wav.unsqueeze(0), n_fft, n_mels, sr, hop_size, win_size, fmin, fmax)
print(x.shape)

f = open('mel.pkl', 'wb')
pickle.dump(x, f)
f.close()
