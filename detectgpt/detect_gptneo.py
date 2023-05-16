import httpx
import msgpack
import json
import random

from tqdm import tqdm


def access_sniffer(text, api_url, language="en", get_data=0, get_dc=0):
    """
    language = "en" or "cn"
    get_data = 0 or 1 or 2
    get_dc = 0 or 1, 2, 3
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

def access_api(text, api_url, do_generate=False):
    """
    :param text:        input text
    :param api_url:     api
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

if __name__ == "__main__":
    with open(
            "",
            'r') as f:
        samples_test = [json.loads(line) for line in f]

    # [values, label_int, label]
    samples_test = [[item['text'], item['label_int'], item['label']]
                    for item in samples_test]
    samples_test = [{
        'text': item[0],
        'label_int': item[1],
        'label': item[2]
    } for item in samples_test]

    processed_sentence = 0
    detect_gptneo_samples = []
    for idx, item in tqdm(enumerate(samples_test)):
        text = item['text']
        label = item['label']
        
        # if label != 'gptj' and label != 'gptneo' and label != 'human':
        if label != 'gpt2' and label != 'gpt3re' and label != 'gpt3sum':
            continue
        assert label == 'gpt2' or label == 'gpt3re' or label == 'gpt3sum', " ERROR: label != 'gpt2' and label != 'gpt3re' and label != 'gpt3sum' "

        t5_url = "http://10.176.52.119:20039/inference"
        url = "http://10.176.52.120:20097/inference"
        ptb_num = 40

        try:
            ptb_texts = []
            for _ in range((ptb_num // 5)):
                ptb_texts.extend(access_api(text, t5_url))
            losses = []
            begin_word_idxes = []
            ll_tokens_list = []
            for ptb_text in ptb_texts:
                loss, begin_word_idx, ll_tokens = access_api(ptb_text, url)
                losses.append(loss)
                begin_word_idxes.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)
            item['losses'] = losses
            item['begin_word_idxes'] = begin_word_idxes
            item['ll_tokens_list'] = ll_tokens_list
            detect_gptneo_samples.append(item)
            processed_sentence += 1
        except:
            print("fail to process this sample, discard it")
            print(idx)

    with open(
            '',
            'w', encoding='utf-8') as f:
        json.dump(detect_gptneo_samples, f)

    print("processed_sentence:", str(processed_sentence))