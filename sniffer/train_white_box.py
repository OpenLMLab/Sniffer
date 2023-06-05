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

project_path = os.path.abspath('project_path')
if project_path not in sys.path:
    sys.path.append(project_path)
import sniffer_model_info


def train(samples_train,
          samples_test,
          model_num=4,
          feat_num=4,
          class_num=5,
          ckpt_name='linear_en.pt',
          train_feat='all',
          output_test_set=False):
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
    linear_model.to('cuda')
    # use all losses and features
    if train_feat == 'all':
        inputs_train = [x[0] for x in samples_train]
        inputs_test = [x[0] for x in samples_test]
    # use loss only
    if train_feat == 'loss_only':
        inputs_train = [x[0][0:model_num] for x in samples_train]
        inputs_test = [x[0][0:model_num] for x in samples_test]
    if train_feat == 'pct-score_only':
        inputs_train = [
            x[0][model_num:model_num + int(model_num * (model_num - 1) / 2)]
            for x in samples_train
        ]
        inputs_test = [
            x[0][model_num:model_num + int(model_num * (model_num - 1) / 2)]
            for x in samples_test
        ]
    if train_feat == 'loss_pct_score':
        inputs_train = [
            x[0][0:model_num + (int)(model_num * (model_num - 1) / 2)]
            for x in samples_train
        ]
        inputs_test = [
            x[0][0:model_num + (int)(model_num * (model_num - 1) / 2)]
            for x in samples_test
        ]
    # use losses and original processed features
    if train_feat == 'ori_feat':
        inputs_train = [
            x[0][0:model_num + (int)(model_num * (model_num - 1) / 2) * 2]
            for x in samples_train
        ]
        inputs_test = [
            x[0][0:model_num + (int)(model_num * (model_num - 1) / 2) * 2]
            for x in samples_test
        ]
    # use all features only
    if train_feat == 'feat_only':
        inputs_train = [x[0][model_num:] for x in samples_train]  # first 3 dim
        inputs_test = [x[0][model_num:] for x in samples_test]

    outputs_train = [x[1] for x in samples_train]
    outputs_test = [x[1] for x in samples_test]

    mask_re = []
    mask_sum = []
    for output in samples_test:
        if output[-1] == 'gpt3re':
            mask_re.append(1)
        else:
            mask_re.append(0)

        if output[-1] == 'gpt3sum':
            mask_sum.append(1)
        else:
            mask_sum.append(0)

    mask_re = torch.tensor(mask_re).to('cuda')
    mask_sum = torch.tensor(mask_sum).to('cuda')

    inputs_train = torch.tensor(inputs_train).to('cuda')
    outputs_train = torch.tensor(outputs_train).to('cuda')

    inputs_test = torch.tensor(inputs_test).to('cuda')
    outputs_test = torch.tensor(outputs_test).to('cuda')

    if output_test_set:
        return inputs_test, outputs_test, mask_re, mask_sum

    # train the linear model
    training(linear_model, inputs_train, outputs_train, inputs_test,
             outputs_test, class_num, [mask_re, mask_sum])

    # save the model and load at detector-backend
    torch.save(linear_model.cpu(), ckpt_name)

    # testing
    saved_model = torch.load(ckpt_name)
    linear_model = linear_model.load_state_dict(saved_model.state_dict())


def training(model, X_train, y_train, X_test, y_test, class_num, masks=[]):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 10000

    num = [sum(y_test == i) for i in range(class_num)]
    mask_re = masks[0].cpu().numpy()
    mask_sum = masks[1].cpu().numpy()
    print(num)

    for it in tqdm(range(n_epochs)):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % 100 == 0:
            with torch.no_grad():
                outputs = model(X_test)
                loss_test = criterion(outputs, y_test)
                prob = torch.nn.functional.softmax(outputs, dim=-1)  # N, label
                pred_labels = torch.argmax(prob, dim=-1)

                true_labels = y_test.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                acc_label = precision_score(y_true=true_labels,
                                            y_pred=pred_labels,
                                            average=None)
                rec_label = recall_score(y_true=true_labels,
                                         y_pred=pred_labels,
                                         average=None)
                acc = (true_labels == pred_labels).astype(
                    np.float32).mean().item()

                rec_label_re = (sum(
                    (pred_labels == true_labels) * mask_re)) / sum(mask_re)
                rec_label_sum = (sum(
                    (pred_labels == true_labels) * mask_sum)) / sum(mask_sum)

                print('*' * 120)
                print(
                    f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}'
                )
                print("Total acc: {}".format(acc))
                print("The accuracy of each class:")
                print(acc_label)
                print("The recall of each class:")
                print(rec_label)
                print("rec_label_re: {}, rec_label_sum: {}".format(
                    rec_label_re, rec_label_sum))
    return


def consturct_train_features(samples_train):
    convert_train = []
    for item in samples_train:
        label = item['label']
        label_int = item['label_int']
        values = item['values']
        features = values['losses'] + values['lt_zero_percents'] + values[
            'std_deviations'] + values['pearson_list'] + values[
                'spearmann_list']
        convert_train.append([features, label_int, label])
    return convert_train


if __name__ == "__main__":
    name = 'train'

    with open(
            '',
            'r') as f:
        samples_train = [json.loads(line) for line in f]
    with open(
            '',
            'r') as f:
        samples_test = [json.loads(line) for line in f]

    # [values, label_int, label]
    samples_train = consturct_train_features(samples_train)
    samples_test = consturct_train_features(samples_test)

    if name == 'train':
        train(
            samples_train=samples_train,
            samples_test=samples_test,
            model_num=4,
            feat_num=4,
            class_num=5,
            ckpt_name=
            '',
            train_feat='all')