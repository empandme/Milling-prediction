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
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 512, "hop_length": 128, "n_mels": 40}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['filepath']
        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(waveform).unsqueeze(0)
        if waveform.shape[1] < 512:
            print(f"Too short {audio_path}")
            return None, None
        mfcc = self.mfcc_transform(waveform)
        mfcc = torch.clamp(mfcc, min=1e-6)
        mfcc = torch.log(mfcc)
        label = torch.tensor(row['Ra'], dtype=torch.float32)
        return mfcc, label

def collate_fn_crop_pad(batch, fixed_len=100):
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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(1)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    count = 0
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
        count += 1
    return total_loss / count if count > 0 else float('inf')

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
            if tmp == patience:
                print(f"Early stopping at epoch {epoch+1}, loss: {loss:.4f}")
                break
        else:
            tmp = 0
        prev_loss = loss
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "surface_roughness_mfcc_cnn.pth")

if __name__ == "__main__":
    main()
