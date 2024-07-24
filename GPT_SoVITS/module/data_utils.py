import logging
import math
import os
import random
import time
import traceback
from functools import lru_cache
from io import BytesIO

import librosa
import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.nn.functional as F
import torch.utils.data
from module import commons
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from scipy.io import wavfile
from text import cleaned_text_to_sequence
from tqdm import tqdm
from utils import load_filepaths_and_text, load_wav_to_torch

# ZeroDivisionError fixed by Tybost (https://github.com/RVC-Boss/GPT-SoVITS/issues/79)
# class TextAudioSpeakerLoader(torch.utils.data.Dataset):
#     """
#     1) loads audio, speaker_id, text pairs
#     2) normalizes text and converts them to sequences of integers
#     3) computes spectrograms from audio files.
#     """

#     def __init__(self, hparams, val=False):
#         exp_dir = hparams.exp_dir
#         self.path2 = "%s/2-name2text.txt" % exp_dir
#         self.path4 = "%s/4-cnhubert" % exp_dir
#         self.path5 = "%s/5-wav32k" % exp_dir
#         assert os.path.exists(self.path2)
#         assert os.path.exists(self.path4)
#         assert os.path.exists(self.path5)
#         names4 = set([name[:-3] for name in list(os.listdir(self.path4))])  # 去除.pt后缀
#         names5 = set(os.listdir(self.path5))
#         self.phoneme_data = {}
#         with open(self.path2, "r", encoding="utf8") as f:
#             lines = f.read().strip("\n").split("\n")

#         for line in lines:
#             tmp = line.split("\t")
#             if (len(tmp) != 4):
#                 continue
#             self.phoneme_data[tmp[0]] = [tmp[1]]

#         self.audiopaths_sid_text = list(set(self.phoneme_data) & names4 & names5)
#         tmp = self.audiopaths_sid_text
#         leng = len(tmp)
#         min_num = 100
#         if (leng < min_num):
#             self.audiopaths_sid_text = []
#             for _ in range(max(2, int(min_num / leng))):
#                 self.audiopaths_sid_text += tmp
#         self.max_wav_value = hparams.max_wav_value
#         self.sampling_rate = hparams.sampling_rate
#         self.filter_length = hparams.filter_length
#         self.hop_length = hparams.hop_length
#         self.win_length = hparams.win_length
#         self.sampling_rate = hparams.sampling_rate
#         self.val = val

#         random.seed(1234)
#         random.shuffle(self.audiopaths_sid_text)

#         print("phoneme_data_len:", len(self.phoneme_data.keys()))
#         print("wav_data_len:", len(self.audiopaths_sid_text))

#         audiopaths_sid_text_new = []
#         lengths = []
#         skipped_phone = 0
#         skipped_dur = 0
#         for audiopath in tqdm(self.audiopaths_sid_text):
#             try:
#                 phoneme = self.phoneme_data[audiopath][0]
#                 phoneme = phoneme.split(' ')
#                 phoneme_ids = cleaned_text_to_sequence(phoneme)
#             except Exception:
#                 print(f"{audiopath} not in self.phoneme_data !")
#                 skipped_phone += 1
#                 continue

#             size = os.path.getsize("%s/%s" % (self.path5, audiopath))
#             duration = size / self.sampling_rate / 2

#             if duration == 0:
#                 print(f"Zero duration for {audiopath}, skipping...")
#                 skipped_dur += 1
#                 continue

#             if 54 > duration > 0.6 or self.val:
#                 audiopaths_sid_text_new.append([audiopath, phoneme_ids])
#                 lengths.append(size // (2 * self.hop_length))
#             else:
#                 skipped_dur += 1
#                 continue

#         print("skipped_phone: ", skipped_phone, ", skipped_dur: ", skipped_dur)
#         print("total left: ", len(audiopaths_sid_text_new))
#         assert len(audiopaths_sid_text_new) > 1  # 至少能凑够batch size，这里todo
#         self.audiopaths_sid_text = audiopaths_sid_text_new
#         self.lengths = lengths

#     def get_audio_text_speaker_pair(self, audiopath_sid_text):
#         audiopath, phoneme_ids = audiopath_sid_text
#         text = torch.FloatTensor(phoneme_ids)
#         try:
#             spec, wav = self.get_audio("%s/%s" % (self.path5, audiopath))
#             with torch.no_grad():
#                 ssl = torch.load("%s/%s.pt" % (self.path4, audiopath), map_location="cpu")
#                 if (ssl.shape[-1] != spec.shape[-1]):
#                     typee = ssl.dtype
#                     ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(typee)
#                 ssl.requires_grad = False
#         except:
#             traceback.print_exc()
#             spec = torch.zeros(1025, 100)
#             wav = torch.zeros(1, 100 * self.hop_length)
#             ssl = torch.zeros(1, 768, 100)
#             text = text[-1:]
#             print("load audio or ssl error!!!!!!", audiopath)
#         return (ssl, spec, wav, text)

#     def get_audio(self, filename):
#         audio_array = load_audio(filename, self.sampling_rate)  # load_audio的方法是已经归一化到-1~1之间的，不用再/32768
#         audio = torch.FloatTensor(audio_array)  # /32768
#         audio_norm = audio
#         audio_norm = audio_norm.unsqueeze(0)
#         spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length,
#                                   center=False)
#         spec = torch.squeeze(spec, 0)
#         return spec, audio_norm

#     def get_sid(self, sid):
#         sid = torch.LongTensor([int(sid)])
#         return sid

#     def __getitem__(self, index):
#         # with torch.no_grad():
#         return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

#     def __len__(self):
#         return len(self.audiopaths_sid_text)


class TextAudioLoader(torch.utils.data.Dataset):
    def tokenize_text(self, text):
        token = [0] + [x + 1 for x in self.sp.encode(text)] + [2]
        return token

    def __init__(self, hparams, val=False):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load("./pretrained_models/sentencepiece.bpe.model")
        exp_dir = hparams.exp_dir
        todo = []
        self.audiopaths_text = []
        self.lengths = []
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file.endswith(".txt"):
                    index_folder = os.path.relpath(root, exp_dir)
                    file_path = os.path.join(root, file)

                    # 尝试不同的编码
                    encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
                    for encoding in encodings:
                        try:
                            with open(file_path, "r", encoding=encoding) as f:
                                lines = f.readlines()
                            break  # 如果成功读取，跳出循环
                        except UnicodeDecodeError:
                            continue  # 如果解码失败，尝试下一个编码
                    else:
                        print(f"无法解码文件 {file_path}，跳过此文件")
                        continue  # 如果所有编码都失败，跳过此文件

                    for line in lines:
                        try:
                            spk_name, wav_name, text = line.split("|")
                            todo.append([spk_name, wav_name, text, index_folder])
                        except Exception:
                            print(line)
        for data in todo:
            _, wav_name, text, index_folder = data
            audio_path = os.path.join(exp_dir, index_folder, wav_name)
            speech_token_path = audio_path + ".npy"
            bert_path = audio_path + ".pt"
            wav_path = audio_path + ".wav"
            if (
                os.path.exists(speech_token_path)
                and os.path.exists(bert_path)
                and os.path.exists(wav_path)
            ):
                try:
                    duration = librosa.get_duration(filename=wav_path)  # noqa: F821
                except Exception as e:
                    print(f"无法处理文件 {wav_path}：{str(e)}")
                    continue
                if duration < 0.7:
                    continue
                self.lengths.append(math.ceil(duration * 50))
                self.audiopaths_text.append([audio_path, text])

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.val = val

        """
        @misc{picard2023torchmanualseed3407needinfluencerandom,
        title={Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision}, 
        author={David Picard},
        year={2023},
        eprint={2109.08203},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2109.08203}, 
        }
        """
        random.seed(3407)  # 3407 is all you need

        random.shuffle(self.audiopaths_text)
        print("wav_data_len:", len(self.audiopaths_text))

    def get_audio_text_speaker_pair(self, audiopath_text):
        audiopath, text = audiopath_text
        text_token = self.tokenize_text(text)
        try:
            spec, wav = self.get_audio(audiopath + ".wav")
            speech_token = np.load(audiopath + ".npy")
            speech_token = torch.from_numpy(speech_token)
            min_length = min(speech_token.shape[-1], spec.shape[-1])
            speech_token = speech_token[..., :min_length]
            spec = spec[..., :min_length]
        except Exception:
            traceback.print_exc()
            spec = torch.zeros(1025, 100)
            wav = torch.zeros(1, 100 * self.hop_length)
            speech_token = torch.zeros(1, 100)
            text_token = text_token[-1:]
            print("load error!!!!!!", audiopath)
        return (speech_token, spec, wav, text_token)

    def get_audio(self, filename):
        audio_array = load_audio(filename, self.sampling_rate)
        audio = torch.FloatTensor(audio_array)
        audio = audio.unsqueeze(0)
        spec = spectrogram_torch(
            audio,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return spec, audio

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        # with torch.no_grad():
        return self.get_audio_text_speaker_pair(self.audiopaths_text[index])

    def __len__(self):
        return len(self.audiopaths_text)


class TextAudioCollate:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # 按照频谱图长度排序
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_speech_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([len(x[3]) for x in batch])

        speech_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        speech_padded = torch.LongTensor(len(batch), max_speech_len)
        spec_padded = torch.FloatTensor(len(batch), 1025, max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)

        speech_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            speech = row[0]
            speech_padded[i, : speech.size(0)] = speech
            speech_lengths[i] = speech.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            text = torch.LongTensor(row[3])
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

        if self.return_ids:
            return (
                speech_padded,
                speech_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                text_padded,
                text_lengths,
                ids_sorted_decreasing,
            )
        return (
            speech_padded,
            speech_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            text_padded,
            text_lengths,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
