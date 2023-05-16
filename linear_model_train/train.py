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

# process

# collect raw data
"""
1. News articles: This can include a variety of topics such as politics, economics, sports, entertainment, etc.
2. Books: Fiction, non-fiction, classic literature, bestsellers, etc.
3. Social media posts: This can include tweets, Facebook posts, Instagram captions, and more.
4. Web pages: Websites of all kinds, including blogs, online magazines, e-commerce sites, and more.
5. Scientific articles: Research papers, studies, scientific reports, and more.
6. Academic papers: Scholarly articles, dissertations, theses, and more.
7. Conversational data: Chat logs, emails, customer service logs, and more.
8. Technical documentation: Manuals, instructions, user guides, and more.
9. Legal documents: Contracts, agreements, court rulings, and more.
10. Creative writing: Short stories, poetry, screenplays, and more.
"""
"""

SQuAD (wiki)
XSUM (BBC News)
IMDB/
webtext
arxiv/ & pubmed





"""


def get_webtext_texts(jsonl_file):
    total_texts = []
    web_texts = open(jsonl_file, 'r', encoding='utf-8').readlines()
    web_texts = [json.loads(text.strip('\n'))['text']
                 for text in web_texts]  # truncate
    web_texts = [text.replace('\n', '') for text in web_texts]
    for text in web_texts:
        tokens = text.split(' ')
        for i in range(0, len(tokens), 512):
            web_text = ' '.join(text.split(' ')[i:i + 512])
            total_texts.append(web_text)
    return total_texts


def get_data():
    pubmed_data = open('en_raw_data/pubmed.txt', 'r',
                       encoding='utf-8').readlines()
    pubmed_data_lines = []
    for data in pubmed_data[:1000]:
        data = json.loads(data)
        abstract = [
            ab.replace("<S>", "").replace("</S>", "").strip()
            for ab in data["abstract_text"]
        ]
        abstract = "  ".join(abstract)
        pubmed_data_lines.append(abstract)
    # print(pubmed_data_lines)

    arxiv_data = open('en_raw_data/arxiv.txt', 'r',
                      encoding='utf-8').readlines()
    arxiv_data_lines = []
    for data in arxiv_data[:1000]:
        data = json.loads(data)
        abstract = [
            ab.replace("<S>", "").replace("</S>", "").strip()
            for ab in data["abstract_text"]
        ]
        abstract = "  ".join(abstract)
        arxiv_data_lines.append(abstract)
    # print(arxiv_data_lines)

    imdb_data = open('en_raw_data/imdb.csv', 'r', encoding='utf-8').readlines()

    imdb_lines = []

    for line in imdb_data[:1000]:
        d = line[1:-10]
        imdb_lines.append(d)

    Xsum_data = open('en_raw_data/xsum.txt', 'r',
                     encoding='utf-8').readlines()[:1000]

    SQuAD_data = open('en_raw_data/squad.json', 'r', encoding='utf-8')
    data = json.load(SQuAD_data)['data']
    squad_data_lines = []
    for doc in data:
        paras = doc['paragraphs']
        for para in paras:
            context = para['context']
            squad_data_lines.append(context)
    squad_data_lines = squad_data_lines[:1000]

    webtext_lines = get_webtext_texts('en_raw_data/webtext.valid.jsonl')[:1000]

    total_lines = webtext_lines + Xsum_data + imdb_lines + squad_data_lines + pubmed_data_lines + arxiv_data_lines

    print('total lines', len(total_lines))

    # total_lines
    random.shuffle(total_lines)

    total_len = len(total_lines)
    test_len = int(total_len / 10)

    # test_lines = total_lines[:test_len]
    # train_lines = total_lines[test_len:]

    fw = open('human_lines.txt', 'w')
    for line in total_lines:
        fw.write(line.replace('\n', ' ') + '\n')


def access_api(text, api_url, do_generate=False):
    """

    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


def get_data_gpts(run_type='gpt2'):
    lines_human = open('human_lines.txt', 'r')

    gpt2_api = 'http://localhost:8001/inference'
    gptj_api = 'http://localhost:8002/inference'
    gptneo_api = 'http://localhost:8003/inference'

    gpt2_lines = []
    gptj_lines = []
    gptneo_lines = []
    if run_type == 'gptj':
        fwname = 'gptj_lines.txt'
    if run_type == 'gptneo':
        fwname = 'gptneo_lines.txt'

    with open(fwname, 'a+', encoding='utf-8') as write_lines:
        for line in tqdm(lines_human):
            tokens = line.split(' ')
            prompt = ' '.join(tokens[:10])
            if run_type == 'gpt2':
                gpt2_text = access_api(prompt, gpt2_api, True)
                if gpt2_text is not None:
                    gpt2_lines.append(gpt2_text)
            if run_type == 'gptj':
                gptj_text = access_api(prompt, gptj_api, True)
                if gptj_text is not None:
                    gptj_lines.append(gptj_text)
                    write_lines.write(gptj_text.replace('\n', ' ') + '\n')

            if run_type == 'gptneo':
                gptneo_text = access_api(prompt, gptneo_api, True)
                if gptneo_text is not None:
                    gptneo_lines.append(gptneo_text)
                    write_lines.write(gptneo_text.replace('\n', ' ') + '\n')

    # if run_type == 'gpt2':
    #     fw = open('gpt2_lines.txt', 'w')
    #     for line in gpt2_lines:
    #         fw.write(line.replace('\n', ' ') + '\n')

    # if run_type == 'gptj':
    #     fw = open('gptj_lines.txt', 'w')
    #     for line in gptj_lines:
    #         fw.write(line.replace('\n', ' ') + '\n')

    # if run_type == 'gptneo':
    #     fw = open('gptneo_lines.txt', 'w')
    #     for line in gptneo_lines:
    #         fw.write(line.replace('\n', ' ') + '\n')


# def get_data_aigc():
# pass


def get_data_aigc_rephrase():
    lines = open('en_gen_data/human_lines.txt', 'r',
                 encoding='utf-8').readlines()
    lines = lines[:2000]
    with open('chatgpt_rephrase.json', 'a+', encoding='utf-8') as write_lines:
        prompts_all = [
            'rephrase the following content:' + line for line in lines
        ]

        for idx in range(0, len(prompts_all), 10):
            print(idx)
            prompts = prompts_all[idx:idx + 10]
            while 1:
                try:
                    model_responses = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompts,
                        max_tokens=1024)
                    model_outputs = [
                        model_response["text"]
                        for model_response in model_responses['choices']
                    ]
                    break
                except openai.error.RateLimitError:
                    print('Rate limit reached, sleep 60 seconds')
                    time.sleep(60)

            for i, l in enumerate(model_outputs):
                write_lines.write(l.replace('\n', '') + '\n')


def get_data_aigc_summ():
    lines = open('en_gen_data/human_lines.txt', 'r',
                 encoding='utf-8').readlines()
    lines = lines[:2000]
    with open('chatgpt_summgen.json', 'a+', encoding='utf-8') as write_lines:
        prompts_all = [
            'summarize the following content:' + line for line in lines
        ]

        for idx in range(0, len(prompts_all), 10):
            print(idx)
            prompts = prompts_all[idx:idx + 10]
            while 1:
                try:
                    model_responses = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompts,
                        max_tokens=1024)
                    model_outputs = [
                        model_response["text"]
                        for model_response in model_responses['choices']
                    ]
                    break
                except openai.error.RateLimitError:
                    print('Rate limit reached, sleep 60 seconds')
                    time.sleep(60)

                # model_outputs = model_outputs
                model_outputs = [
                    line.replace('summarize the following content:', '')
                    for line in model_outputs
                ]
                prompts = [
                    'write a document based on the following content:' + line
                    for line in model_outputs
                ]

                try:
                    model_responses = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompts,
                        max_tokens=1024)
                    model_outputs = [
                        model_response["text"]
                        for model_response in model_responses['choices']
                    ]
                    break
                except openai.error.RateLimitError:
                    print('Rate limit reached, sleep 60 seconds')
                    time.sleep(60)

            for i, l in enumerate(model_outputs):
                write_lines.write(l.replace('\n', '') + '\n')


def access_sniffer(text, api_url, language="en", get_data=0, get_dc=0):
    """
    language = "en" or "cn"
    get_data = 0 or 1 or 2
    get_dc = 0 or 1
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "language": language,
            "get_data": get_data,
            "get_dc": get_dc,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        print(prediction)
        content = None
    return content


def get_lines():
    human_lines = open('en_gen_data/human_lines.txt', 'r').readlines()
    human_lines = [[line, 'human'] for line in human_lines]

    gpt2_lines = open('en_gen_data/gpt2_lines.txt', 'r').readlines()

    # post-process gpt2 lines
    gpt2_lines = [line.replace('<|endoftext|>', '') for line in gpt2_lines]
    gpt2_lines = [[line, 'gpt2'] for line in gpt2_lines]

    gptj_lines = open('en_gen_data/gptj_lines.txt', 'r').readlines()

    gptj_lines = [line.replace('<|endoftext|>', '') for line in gptj_lines]
    gptj_lines = [[line, 'gptj'] for line in gptj_lines]

    # post-process  lines

    gptneo_lines = open('en_gen_data/gptneo_lines.txt', 'r').readlines()

    gptneo_lines = [line.replace('<|endoftext|>', '') for line in gptneo_lines]
    gptneo_lines = [[line, 'gptneo'] for line in gptneo_lines]

    # post-process  lines

    gpt3_re_lines = open('en_gen_data/chatgpt_rephrase.json', 'r').readlines()
    gpt3_re_lines = [
        line.replace('rephrase the following content:', '')
        for line in gpt3_re_lines
    ]
    gpt3_re_lines = [[line, 'gpt3re'] for line in gpt3_re_lines]

    gpt3_sum_lines = open('en_gen_data/chatgpt_summgen.json', 'r').readlines()
    gpt3_sum_lines = [
        line.replace('summarize the following content:', '')
        for line in gpt3_sum_lines
    ]
    gpt3_sum_lines = [[line, 'gpt3sum'] for line in gpt3_sum_lines]

    # post-process

    labels = {'gpt2': 0, 'gpteai': 1, 'gpt3': 2, 'llama': 3, 'human': 4}

    # gpt3 (rephrase & summ)

    api_url = "http://localhost:7999/inference"
    # API get data

    total_lines = gpt3_re_lines + gpt3_sum_lines + gptneo_lines + gptj_lines + gpt2_lines + human_lines
    random.shuffle(total_lines)

    length = len(total_lines)
    print('total samples', length)

    test_length = int(length / 10)

    train_lines = total_lines[:-test_length]

    test_lines = total_lines[-test_length:]

    json.dump(train_lines,
              open('en_features/trains.json', 'w', encoding='utf-8'),
              ensure_ascii=False)

    json.dump(test_lines,
              open('en_features/tests.json', 'w', encoding='utf-8'),
              ensure_ascii=False)


def get_features(language, sniffer_api):
    """ You need to modify four filename and the api_url """
    total_features = []

    train_lines = json.load(
        open('en_features/trains.json', 'r', encoding='utf-8'))
    test_lines = json.load(
        open('en_features/tests.json', 'r', encoding='utf-8'))

    total_lines = train_lines + test_lines
    train_length = len(train_lines)

    if language == 'en':
        labels = sniffer_model_info.en_labels
    else:
        labels = sniffer_model_info.cn_labels

    for data in tqdm(total_lines):
        text, label = data
        label_int = labels[label]

        result = access_sniffer(text, sniffer_api, get_data=2)
        if result is not None:
            _, _, _, values = result
            total_features.append([values, label_int, label])
        else:
            print('error')

    train_features = total_features[:train_length]
    test_features = total_features[train_length:]

    json.dump(train_features,
              open('en_features/trains_features.json', 'w', encoding='utf-8'))
    json.dump(test_features,
              open('en_features/tests_features.json', 'w', encoding='utf-8'))


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
        label_int = item['label_int']
        label = item['label']
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
    import random
    random.shuffle(samples_train)
    train_len = len(samples_train) // 1
    samples_train = samples_train[0:train_len]
    samples_test = consturct_train_features(samples_test)

    if name == 'train':
        train(
            samples_train=samples_train,
            samples_test=samples_test,
            model_num=len(sniffer_model_info.en_model_names),
            feat_num=sniffer_model_info.cur_feat_num,
            class_num=5,
            ckpt_name=
            '',
            train_feat='all')