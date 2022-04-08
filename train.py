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


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print("Loss: {}".format(loss.item()))


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print("Epochs {}".format(i + 1))
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("----------------------")
    print("Training is done")


if __name__ == "__main__":

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

    # create a data loader for the train dataset
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using {}".format(device))
    cnn = CNN().to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device,
          EPOCHS)

    torch.save(cnn.state_dict(), SAVE_PATH)
    print("Model trained and stored at CNN_audio_classification.pth")
