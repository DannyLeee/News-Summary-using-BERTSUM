
import pandas as pd
import os
from transformers import BertTokenizer
import torch
import xml.etree.cElementTree as ET

tree = ET.parse('./dataset/TC/TestFile_TD_DOS/GoldHair_DocRatio_REF_20Test_Text01.xml')
root = tree.getroot()
test_set = set()
for child in root:
    test_set.add(child[3][0].text)

LM = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(LM)

stories_dir = os.path.abspath("./dataset/TC/TestFile_TD_DOS")
preporcessed_stories_dir = os.path.abspath("./dataset/bert_data_for_BERTSUM/TC")

"""dictionary"""
DICT_PATH = "./dataset/TC/Lexicon2003-72k.txt"
df_dict = pd.read_csv(DICT_PATH, encoding="cp950", header=None)

print("Preparing to preporcess %s to %s ..." % (stories_dir, preporcessed_stories_dir))
stories = os.listdir(stories_dir)
# make IO list file
print("Making list of files to preporcess...")

for s in stories:
    if (not s.endswith('txt')):
        stories.remove(s)

Human_N = '1'
Percent = {10:'01', 20:'02', 30:'03'}

"""
answer decoder
input: index sequence (list of string)
output: decode of index (string)
"""
def decoder(index):
    result = ""
    for i in index:
        if (int(i)>8533):
            print(df_dict.loc[int(i), 0])
        result += df_dict.loc[int(i), 0]
    return result

"""
content preprocess
input: content (string), answer (string)
output: processed content(string), label (list of 1/0), cls id (list of int),
        origin content (list of string), target text (list of string)
"""
def content_preprocess(content, ans):
    content = content.replace(" ", "")
    content_list = content.splitlines()
    origin_content = content_list[:] # create a shallow copy, if don't add [:] will get the pointer
    result_label = []
    segments_ids = []
    cls_ids = []
    tgt_text_list = []
    pos = 0
    for i, text in enumerate(content_list):
        if (pos < 512):
            cls_ids.append(pos)
            pos += len(text) + 2 # +2 for [CLS], [SEP]
            if (ans.find(text) != -1):
                result_label.append(1) # label
                tgt_text_list.append(text)
            else:
                result_label.append(0) # label
        content_list[i] = "[CLS] " + text + " [SEP]"
    result_content = ' '.join(content_list)
    tgt_text = '<q>'.join(tgt_text_list)
    return result_content, result_label, cls_ids, origin_content, tgt_text

"""
segement embedding
code from BERTSUM
input: input_ids (list of int)
output: segment embedding (list of int)
"""
def create_segment(input_ids):
    segments_ids = []
    _segs = [-1] + [i for i, t in enumerate(input_ids) if t == tokenizer.vocab['[SEP]']]
    segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
    for i, s in enumerate(segs):
        if (i % 2 == 0):
            segments_ids += s* [0]
        else:
            segments_ids += s * [1]
    return segments_ids

train_dataset = []
test_dataset = []

for i, FILE_NAME in enumerate(stories):
    """answer"""
    ANS_PATH = "./dataset/TC/TestFile_TD_DOS/DocRatio/Human_" + Human_N + "/" + Percent[10] + "/" + FILE_NAME
    ans_file = open(ANS_PATH)
    raw_ans = ans_file.read()

    raw_ans = raw_ans.replace("a","").replace("\n", "")
    ans = raw_ans.split(" ")
    ans = ans[:-1]
    ans = decoder(ans)

    """news"""
    NEWS_PATH = "./dataset/TC/TestFile_TD_DOS/" + FILE_NAME
    news_file = open(NEWS_PATH, encoding="cp950")
    news = news_file.read()

    content, label, cls_ids, origin_content, tgt_text = content_preprocess(news, ans)

    bert_dict = tokenizer.encode_plus(content,
                                    add_special_tokens = False,
                                    return_token_type_ids = False,
                                    max_length=512,
                                    pad_to_max_length=True,
                                    return_tensors='pt',
                                    truncation=True)

    bert_dict['input_ids'][0][511] = tokenizer.vocab['[SEP]']
    segments_ids = create_segment(bert_dict['input_ids'][0])

    data_dict = {"src": bert_dict['input_ids'][0].tolist(), "segs": segments_ids, "att_msk" : bert_dict['attention_mask'][0].tolist(), "labels": label,  'clss': cls_ids,
                        'src_txt': origin_content, "tgt_txt": tgt_text}

    if FILE_NAME in test_set:
        test_dataset.append(data_dict)
    else:
        train_dataset.append(data_dict)


# torch.save(train_dataset, (preporcessed_stories_dir + "/PTS_all.train.pt"))
# torch.save(test_dataset,  (preporcessed_stories_dir + "/PTS_all.test.pt"))

print("Preprocess Done!")