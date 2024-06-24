import time

import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import logging



from transformers import Wav2Vec2FeatureExtractor, HubertModel, Wav2Vec2BertModel, AutoFeatureExtractor

import utils
import torch.nn as nn

logging.getLogger("numba").setLevel(logging.WARNING)
cnhubert_base_path = None


class Hubert(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = HubertModel.from_pretrained(cnhubert_base_path, cache_dir='./pretrained_models')
        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        #     cnhubert_base_path
        # )
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", cache_dir='./pretrained_models')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0",
            cache_dir='./pretrained_models'
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        )['input_values'].to(x.device)
        feats = self.model(input_values).last_hidden_state
        return feats

def get_model():
    model = Hubert()
    model.eval()
    return model

def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1, 2)


if __name__ == "__main__":
    model = get_model()
    src_path = "/Users/Shared/原音频2.wav"
    wav_16k_tensor = utils.load_wav_to_torch_and_resample(src_path, 16000)
    model = model
    wav_16k_tensor = wav_16k_tensor
    feats = get_content(model, wav_16k_tensor)
    print(feats.shape)
