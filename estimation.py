import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import socket

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(1)

model = CNNRegression().to(DEVICE)
model.load_state_dict(torch.load("surface_roughness_cnn.pth", map_location=DEVICE))
model.eval()

stft_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128)

UDP_IP = "255.255.255.255"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def preprocess_audio(audio, sr, fixed_len=1025, expected_freq=257):
    waveform = torch.tensor(audio).unsqueeze(0)
    spec = stft_transform(waveform)
    spec = torch.log(spec + 1e-6)

    #时间维度检查
    length = spec.shape[2]
    if length > fixed_len:
        spec = spec[:, :, :fixed_len]
    elif length < fixed_len:
        pad_size = fixed_len - length
        pad = torch.zeros(spec.shape[0], spec.shape[1], pad_size)
        spec = torch.cat([spec, pad], dim=2)

    #频率维度检查
    if spec.shape[1] != expected_freq:
        if spec.shape[1] > expected_freq:
            spec = spec[:, :expected_freq, :]
        else:
            pad_size = expected_freq - spec.shape[1]
            pad = torch.zeros(spec.shape[0], pad_size, spec.shape[2])
            spec = torch.cat([spec, pad], dim=1)

    if spec.ndim == 3:
        spec = spec.unsqueeze(0)

    return spec.to(DEVICE)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata[:, 0]
    sr = 44100
    x = preprocess_audio(audio, sr)
    with torch.no_grad():
        pred = model(x).item()
    print(f"Predicted Ra: {pred:.4f}")
    message = f"Ra_prediction:{pred:.4f}"
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

def main():
    sr = 44100
    block_duration = 0.5
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr*block_duration)):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()