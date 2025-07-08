import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa


CSV_PATH = "./audio_labels.csv"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SurfaceRoughnessAudioDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.stft_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['filepath']
        waveform,sr = librosa.load(audio_path)
        waveform=torch.tensor(waveform).unsqueeze(0)
        if waveform.shape[1] < 512:
            print(f"Too short {audio_path}")
            return None, None
        spec = self.stft_transform(waveform)  # [1, freq, time]
        spec = torch.log(spec + 1e-6)         # Log scale improves learning

        label = torch.tensor(row['Ra'], dtype=torch.float32)
        return spec, label

def collate_fn_crop_pad(batch, fixed_len=1025):
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return None, None  
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    processed_inputs = []
    for tensor in inputs:
        length = tensor.shape[2]
        if length > fixed_len:
            cropped = tensor[:, :, :fixed_len]
        elif length < fixed_len:
            # 不够长，补零
            pad_size = fixed_len - length
            pad = torch.zeros(tensor.shape[0], tensor.shape[1], pad_size)
            cropped = torch.cat([tensor, pad], dim=2)
        else:
            cropped = tensor
        processed_inputs.append(cropped)
    
    inputs_tensor = torch.stack(processed_inputs)
    labels_tensor = torch.tensor(labels)
    return inputs_tensor, labels_tensor


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
            nn.Linear(128, 1)  # Regression output
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(1)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        if x is None or y is None:
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    patience = 3
    minichange = 1e-4
    prev_loss = float('inf')
    tmp = 0
    dataset = SurfaceRoughnessAudioDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_crop_pad)   
    model = CNNRegression().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(40):
        loss = train(model, loader, optimizer, criterion)
        if abs(loss - prev_loss) < minichange:
            tmp += 1
            if tmp == patience and abs(loss - prev_loss) < minichange:
                print(f"Early stopping at epoch {epoch+1}, loss: {loss:.4f}")
                break
        prev_loss = loss
        print(f"Epoch {epoch+1}/{20}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "surface_roughness_cnn.pth")

if __name__ == "__main__":
    main()