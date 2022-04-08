import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
from CNN import CNN


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_PATH = "/UrbanSoundClassifier/CNN_audio_classification.pth"
ANNOTATIONS_FILE = "/datasets/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/datasets/UrbanSound8K/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

usd = UrbanSoundDataset(
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    NUM_SAMPLES
)


train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(),
                             lr=LEARNING_RATE)


if __name__ == "__main__":

    for i in range(EPOCHS):
        print("Epochs {}".format(i + 1))
        for inputs, targets in train_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = cnn(inputs)
            loss = loss_fn(predictions, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print("Loss: {}".format(loss.item()))
        print("----------------------")
    print("Training is done")

    torch.save(cnn.state_dict(), SAVE_PATH)
    print("Model trained and stored at CNN_audio_classification.pth")
