from transformers import Wav2Vec2FeatureExtractor
from transformers import RobertaTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import soundfile as sf
import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import os

emos = ["hap", 'neu', 'sad', 'ang']
allemos = ["happy", 'neutral', 'sad', 'angry']
gens = ["F", "M"]
sess = ['session01', 'session02', 'session03', 'session04', 'session05']
emo2idx, idx2emo = {}, {}
idx2allemo = {}
for ii, emo in enumerate(allemos): idx2allemo[ii] = emo
for ii, emo in enumerate(emos): emo2idx[emo] = ii
for ii, emo in enumerate(emos): idx2emo[ii] = emo

gen2idx, idx2gen = {}, {}
for ii, gen in enumerate(gens): gen2idx[gen] = ii
for ii, gen in enumerate(gens): idx2gen[ii] = gen

idx2sess = {}
for ii, ses in enumerate(sess): idx2sess[str(ii+1)] = ses


def read_text(text_path):
    f = open(text_path, "r")
    lines = f.readlines()
    f.close()
    return lines
    
@dataclass
class MultiTaskDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature_extractor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_features, emo_label, video_input, alen, vlen = [], [], [], [], []
        sample_rate = self.feature_extractor.sampling_rate
        batch = {}

        for feature in features:
            input_features.append(feature[0])
            video_input.append(feature[1])
            emo_label.append(feature[2])
            alen.append(feature[3])
            vlen.append(feature[4])

        # batch = self.feature_extractor.pad(
        #     input_features,
        #     padding=self.padding,
        #     max_length=self.max_length * sample_rate,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     truncation=True,
        #     return_tensors="pt",
        # )

        d_type = torch.long if isinstance(emo_label[0], int) else torch.float32

        batch['input_values'] = torch.stack(input_features, dim=0)
        batch["emo_labels"] = torch.tensor(emo_label, dtype=d_type)
        batch["vinput_values"] = torch.stack(video_input, dim=0)

        vframe = torch.tensor(vlen).to(torch.long)
        batch_size = len(vlen)
        v_atten_mask = torch.zeros((batch_size, 180), dtype=torch.long)
        v_atten_mask[(torch.arange(v_atten_mask.shape[0]), vframe - 1)] = 1
        v_atten_mask = v_atten_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        batch['vattention_mask'] = v_atten_mask

        vframe = torch.tensor(alen).to(torch.long)
        attention_mask = torch.zeros((batch_size, 6*16000), dtype=torch.long)
        attention_mask[(torch.arange(attention_mask.shape[0]), vframe - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1])

        batch["attention_mask"] = attention_mask

        return batch

class MultiTaskDataset(Dataset):

    def __init__(self, src_path):
        all_lines = read_text(src_path)
        self.label = []
        self.wav_list = []
        self.visual_list = []
        self.gen_label = []
        self.spk_dict = {}
        self.visual_path = "/home/lqf/workspace/PhD-project/dataset-process/clip-large/"
        for line in all_lines:
            tmp = line.strip().split("\n")[0].split()
            # spk_info = tmp[1].split("/")[-1].split(".")[0].split("_")
            name = tmp[1].split("/")[-1].split(".")[0]
            vpath = os.path.join(self.visual_path, idx2sess[tmp[0][4]], name+".npy")
            self.visual_list.append(vpath)
            self.wav_list.append(tmp[1])
            self.label.append(emo2idx[tmp[-1]])
        
    def __getitem__(self, index):

        wave, sr = sf.read(self.wav_list[index])
        assert sr == 16000
        
        a_len = 0
        if len(wave) >= 6 * 16000:
            wave = wave[:6*16000]
            a_len = 6*16000
        else:
            a_len = len(wave)
            num_len = 6*16000 - a_len
            wave = np.pad(wave, ((0, num_len)), "constant")

        
        vdata = np.load(self.visual_list[index])

        #padding
        v_len = 0
        if len(vdata) >= 180:
            vdata = vdata[:180]
            v_len = 180
        else:
            v_len = len(vdata)
            num_len = 180 - v_len
            vdata = np.pad(vdata, ((0, num_len), (0, 0)), "constant")

        
        lab = self.label[index]

        return torch.FloatTensor(wave), torch.FloatTensor(vdata), lab, a_len, v_len
    

    def __len__(self):
        return len(self.label)
    
    def class_weight_v(self):
        labels = np.array(self.label)
        class_weight = torch.tensor([1/x for x in np.bincount(labels)], dtype=torch.float32)
        return class_weight
    
    def class_weight_q(self):
        class_weight = self.class_weight_v()
        return class_weight / class_weight.sum()
    
    def class_weight_k(self):
        labels = np.array(self.label)
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        weight = weight.tolist()
        samples_weight = torch.tensor([weight[t] for t in labels], dtype=torch.float32)
        """
        class_sample_count = np.unique(labels, return_counts=True)[1]
        class_sample_count = class_sample_count / len(label)
        weight = 1 / class_sample_count
        """
        return samples_weight
    
    def class_weight(self):
        self.emos = Counter(self.label)
        self.emoset = [0,1,2,3]
        weights = torch.tensor([self.emos[c] for c in self.emoset]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()

        return weights


def get_loaders(args, train_path, valid_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/lqf/workspace/wavlm-multi/wavlm-large", return_attention_mask=True)
    data_collator = MultiTaskDataCollator(feature_extractor=feature_extractor, padding=True, max_length=args.max_length, max_length_labels=32)

    train_dataset = MultiTaskDataset(train_path)
    class_weight = train_dataset.class_weight_k()
    valid_dataset = MultiTaskDataset(valid_path)
    # test_dataset = MERDataset(test_path)
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, sampler=sampler, collate_fn=data_collator, num_workers=10)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator, num_workers=10)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)

    return train_dataloader, valid_dataloader, class_weight
    
if __name__ == "__main__":
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/lqf/workspace/wavlm-multi/wavlm-large", return_attention_mask=True)
    # tokenizer = RobertaTokenizer.from_pretrained('/home/lqf/workspace/roberta-base')
    data_collator = MultiTaskDataCollator(feature_extractor=feature_extractor, padding=True, max_length=6)

    train_dataset = MultiTaskDataset("/home/lqf/workspace/icassp2023/session1_test.scp")
    class_weight = train_dataset.class_weight_k()
    sampler = WeightedRandomSampler(weights=class_weight, num_samples=train_dataset.__len__())
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8, sampler=sampler, collate_fn=data_collator)

    for batch in train_dataloader:
        print(batch['attention_mask'])

    pass
