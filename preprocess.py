
import pandas as pd
import os
from transformers import AutoTokenizer
import torch
import xml.etree.cElementTree as ET
import argparse 

"""
answer decoder
input: index sequence (list of string)
output: decode of index (string)
"""
def decoder(index, df_dict):
    result = ""
    for i in index:
        result += df_dict.loc[int(i), 0]
    return result

"""
content preprocess
input: content (string), answer (string)
output: processed content(string), label (list of 1/0), cls id (list of int),
        origin content (list of string), target text (list of string)
"""
def content_preprocess(tokenizer, content, ans="", LM="LM/chinese_wwm_pytorch"):
    content = content.replace(" ", "")
    content_list = content.splitlines()
    origin_content = content_list[:] # create a shallow copy, if don't add [:] will get the pointer
    result_label = []
    segments_ids = []
    cls_ids = []
    tgt_text_list = []
    pos = 0
    for i, text in enumerate(content_list):
        if LM.find("xlnet") == -1: # BERT/RoBERTa
            cls_ids.append(pos)
            pos += len(text) + 2 # +2 for [CLS], [SEP]
        else: # xlnet
            pos += len(text) + 2 # +2 for [CLS], [SEP]
            cls_ids.append(pos)

        if (ans.find(text) != -1):
            result_label.append(1) # label
            tgt_text_list.append(text)
        else:
            result_label.append(0) # label
        
        # truncate to 512
        result_label = [i for i in result_label if i <= 511]
        cls_ids = [i for i in cls_ids if i <= 511]

        if LM.find("xlnet") == -1: # BERT/RoBERTa
            content_list[i] = tokenizer.cls_token + text + tokenizer.sep_token # [CLS] sentence [SEP]
        else: # xlnet
            content_list[i] = text + tokenizer.sep_token + tokenizer.cls_token # sentence <sep> <cls>
    result_content = ' '.join(content_list)
    tgt_text = '<q>'.join(tgt_text_list)
    return result_content, result_label, cls_ids, origin_content, tgt_text

"""
segement embedding
code from BERTSUM
input: input_ids (list of int)
output: segment embedding (list of int)
"""
def create_segment(tokenizer, input_ids, LM="LM/chinese_wwm_pytorch"):
    segments_ids = []
    if LM.find("xlnet") == -1: # BERT/RoBERTa
        _segs = [-1] + [i for i, t in enumerate(input_ids) if t == tokenizer.sep_token_id]
    else: # xlnet
        _segs = [-1] + [i for i, t in enumerate(input_ids) if t == tokenizer.cls_token_id]
    segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
    for i, s in enumerate(segs):
        if (i % 2 == 0):
            segments_ids += s* [0]
        else:
            segments_ids += s * [1]
    return segments_ids

def get_bert_data(news, ans="", mode="bertsum", LM="LM/chinese_wwm_pytorch"):
    tokenizer = AutoTokenizer.from_pretrained(LM)
    content, label, cls_ids, origin_content, tgt_text = content_preprocess(tokenizer, news, ans, LM)

    bert_dict = tokenizer.encode_plus(content,
                                    add_special_tokens = False,
                                    return_token_type_ids = None,
                                    max_length=512,
                                    padding='max_length',
                                    return_tensors='pt',
                                    truncation=True)

    if LM.find("xlnet") == -1: # BERT
        bert_dict['input_ids'][0][511] = tokenizer.sep_token_id
    else: # xlnet
        bert_dict['input_ids'][0][510] = tokenizer.sep_token_id
        bert_dict['input_ids'][0][511] = tokenizer.cls_token_id
    segments_ids = create_segment(tokenizer, bert_dict['input_ids'][0])

    if LM.find("xlnet") != -1: # xlnet
        cls_ids = []
        for _, ids in enumerate(bert_dict["input_ids"][0]):
            if ids == tokenizer.cls_token_id:
                cls_ids += [_]
    label = label[: len(cls_ids)]

    if (mode == "bertsum"):
        data_dict = {"src": bert_dict['input_ids'][0].tolist(), "segs": segments_ids, "att_msk" : bert_dict['attention_mask'][0].tolist(), \
                     "labels": label,  'clss': cls_ids, 'src_txt': origin_content, "tgt_txt": tgt_text}
    elif (mode == "gan"):
        data_dict = {"src": bert_dict['input_ids'][0].tolist(), "segs": segments_ids, "att_msk" : bert_dict['attention_mask'][0].tolist(), \
                     'src_txt': origin_content, "tgt_txt": tgt_text}
    return data_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-LM", default="LM/chinese_wwm_pytorch", type=str)
    parser.add_argument("-result_path", default="./dataset/for_BERTSUM", type=str, help="result folder path")
    parser.add_argument("-human_N", default="1", type=str)
    parser.add_argument("-percent", default="01", type=str, choices=["01", "02", "03"])
    parser.add_argument("-mode", default="bertsum", type=str)
    args = parser.parse_args()

    tree = ET.parse('./dataset/TC/TestFile_TD_DOS/GoldHair_DocRatio_REF_20Test_Text01.xml')
    root = tree.getroot()
    test_set = set()
    for child in root:
        test_set.add(child[3][0].text)

    stories_dir = os.path.abspath("./dataset/TC/TestFile_TD_DOS")

    """dictionary"""
    DICT_PATH = "./dataset/TC/Lexicon2003-72k.txt"
    df_dict = pd.read_csv(DICT_PATH, encoding="cp950", header=None)

    print("Preparing to preporcess %s to %s ..." % (stories_dir, args.result_path))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to preporcess...")

    for s in stories:
        if (not s.endswith('txt')):
            stories.remove(s)

    train_dataset = []
    test_dataset = []

    for i, FILE_NAME in enumerate(stories):
        """answer"""
        ANS_PATH = "./dataset/TC/TestFile_TD_DOS/DocRatio/Human_" + args.human_N + "/" + args.percent + "/" + FILE_NAME
        ans_file = open(ANS_PATH)
        raw_ans = ans_file.read()

        raw_ans = raw_ans.replace("a","").replace("\n", "")
        ans = raw_ans.split(" ")
        ans = ans[:-1]
        ans = decoder(ans, df_dict)

        """news"""
        NEWS_PATH = "./dataset/TC/TestFile_TD_DOS/" + FILE_NAME
        news_file = open(NEWS_PATH, encoding="cp950")
        news = news_file.read()

        data_dict = get_bert_data(news, ans, args.mode, args.LM)

        if FILE_NAME in test_set:
            test_dataset.append(data_dict)
        else:
            train_dataset.append(data_dict)


    torch.save(train_dataset, (args.result_path + "/PTS_all.train.pt"))
    torch.save(test_dataset,  (args.result_path + "/PTS_all.test.pt"))

    print("Preprocess Done!")

if __name__ == '__main__':
    main()