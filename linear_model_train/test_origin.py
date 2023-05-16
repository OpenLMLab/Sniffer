import json
import random
import httpx
import msgpack
import numpy as np
import openai
import time
import torch

from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

import os
import sys
import json

project_path = os.path.abspath('project_path')
if project_path not in sys.path:
    sys.path.append(project_path)
import sniffer_model_info


def test(samples_test,
         model_num=4,
         feat_num=4,
         class_num=5,
         ckpt_name='linear_en.pt',
         train_feat='all'):
    # model_num `loss`s  and  C(model_num, 2) `feature`s
    hid_dim = model_num + int(model_num * (model_num - 1) / 2) * feat_num
    # hid_dim = model_num
    # hid_dim = int(model_num * (model_num - 1) / 2)
    # hid_dim = model_num + int(model_num * (model_num - 1) / 2)
    linear_model = torch.nn.Sequential(torch.nn.Linear(hid_dim, 300),
                                       torch.nn.Dropout(0.5), torch.nn.ReLU(),
                                       torch.nn.Linear(300, 300),
                                       torch.nn.Dropout(0.5), torch.nn.ReLU(),
                                       torch.nn.Linear(300, class_num))
    saved_model = torch.load(ckpt_name)
    linear_model.load_state_dict(saved_model.state_dict())
    linear_model.eval()
    linear_model.to('cuda')
    model = linear_model
    # use all losses and features
    if train_feat == 'all':
        inputs_test = [x[0] for x in samples_test]

    inputs_test = torch.tensor(inputs_test).to('cuda')

    with torch.no_grad():
        outputs = model(inputs_test)
        prob = torch.nn.functional.softmax(outputs, dim=-1)  # N, label
        prob = prob.cpu().numpy()
        prob = prob.tolist()
        with open(
            '',
            'w', encoding='utf-8') as f:
            json.dump(prob, f)
    return


def consturct_train_features(samples_train):
    convert_train = []
    for item in samples_train:
        label_int = item['label_int']
        label = item['label']
        values = item['values']
        features = values['losses'] + values['lt_zero_percents'] + values[
            'std_deviations'] + values['pearson_list'] + values[
                'spearmann_list']
        convert_train.append([features, label_int, label])
    return convert_train


if __name__ == "__main__":
    name = 'test'

    with open(
            'test_data_path',
            'r') as f:
        samples_test = [json.loads(line) for line in f]

    # [values, label_int, label]
    print(len(samples_test))
    samples_test = consturct_train_features(samples_test)
    samples_test = [sample for sample in samples_test if sample[1] == 1]

    if name == 'test':
        test(
            samples_test=samples_test,
            model_num=len(sniffer_model_info.en_model_names),
            feat_num=sniffer_model_info.cur_feat_num,
            class_num=sniffer_model_info.en_class_num,
            ckpt_name=
            'checkpoint_path',
            train_feat='all')
