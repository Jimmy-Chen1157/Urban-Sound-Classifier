import torch
import torchaudio
from CNN import CNN
from urbansounddataset import UrbanSoundDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, SAVE_PATH

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1,10) -> [[0.1, 0.01, ...., 0.6]]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNN()
    state_dict = torch.load(SAVE_PATH)
    cnn.load_state_dict(state_dict)

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

    # get a sample from the validation dataset for inference
    input, target = usd[0][0], usd[0][1]
    input.unsqueeze_(0)

    # make inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print("predicted: {}, Expected: {}".format(predicted, expected))
