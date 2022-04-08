import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotation_file,
                 audio_dir,
                 transform,
                 target_sample_rate,
                 num_samples):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_path(index)
        label = self.get_audio_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self.resample_if_necessary(signal, sr)
        signal = self.mix_down_if_necessary(signal)
        signal = self.cut_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)
        signal = self.transform(signal)
        return signal, label

    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing = self.num_samples - length_signal
            padding = (0, num_missing)
            signal = F.pad(signal, padding)
        return signal

    def resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def get_audio_label(self, index):
        return self.annotations.iloc[index, 6]

    def get_audio_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold,
                            self.annotations.iloc[index, 0])
        return path


if __name__ == "__main__":
    annotations_file = "/datasets/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir = "/datasets/UrbanSound8K/UrbanSound8K/audio"
    sample_rate = 22050
    num_samples = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(
        annotations_file,
        audio_dir,
        mel_spectrogram,
        sample_rate,
        num_samples
    )

    signal, label = usd[1]
    print(signal.shape)
