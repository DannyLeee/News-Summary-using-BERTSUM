from preprocess import get_bert_data
from header import timestamp
import torch
import glob
import os
from multiprocessing import Pool
import json
from opencc import OpenCC

LM = "LM/chinese_wwm_pytorch/"
en2encoder = {"cls":"classifier","trans":"transformer", "rnn":"rnn"}

def get_summary(mode, encoder, origin_text):
    MODEL = en2encoder[encoder]
    
    cc = OpenCC('tw2sp')
    if mode == "abs":
        origin_text = cc.convert(origin_text)

    # preprocess to bert_data
    timestamp("preprocess to bert_data")
    bert_data = get_bert_data(origin_text, LM=LM)
    bert_data["tgt"] = []
    if mode == "abs":
        for i, ids in enumerate(bert_data["src"]):
            bert_data["src"][i] = ids+1 if ids!=0 else 0
    torch.save([bert_data], f"dataset/inference/{mode}_infer.test.pt")

    # feed into model by calling command
    BERT_DATA_PATH = f"./dataset/inference/{mode}_infer"
    RESULT_PATH = f"./result/inference/infer_{mode}_{MODEL}"
    VISIBLE_GPUS = "0"
    GPU_RANKS = "0"
    timestamp("feed to model")
    if mode == "ext":
        command = f"python3 /home/danny/BertSum/src/train.py -mode test -report_rouge false -bert_data_path {BERT_DATA_PATH} -visible_gpus {VISIBLE_GPUS} -gpu_ranks {GPU_RANKS} -batch_size 3000 -decay_method noam -log_file ./logs/inference -use_interval true -temp_dir ./temp -result_path {RESULT_PATH} -rnn_size 768 -bert_config_path {LM}/config.json -test_from ../BertSum/models/NewsSummary/PTS/bert_{MODEL}/model_step_180.pt -encoder {MODEL}" # TODO: can change model?
    elif mode == "abs":
        #  min_length = max(int(len(origin_text) * 0.05), 5)
        min_length = 10
        max_length = max(int(len(origin_text) * 0.3), 10)
        step = 1000000
        command = f"python3 /home/B10615023/PreSummWWM/src/train.py -task abs -mode test -report_rouge false -block_trigram False -batch_size 3000 -test_batch_size 500 -bert_data_path {BERT_DATA_PATH} -log_file ./logs/inference -sep_optim true -use_interval true -visible_gpus {VISIBLE_GPUS} -gpu_ranks {GPU_RANKS} -max_pos 512 -alpha 0.97 -result_path {RESULT_PATH} -min_length {min_length} -max_length {max_length} -beam_size 10 -test_from /home/B10615023/PreSummWWM/model/model_step_{step}.pt"
    flag = os.system(command)
    if flag != 0:
        return f"Error: command failed; err_code: {flag}"

    # get output
    if mode == "abs":
        list_of_files = glob.glob('./result/inference/infer_abs*.candidate') # * means all if need specific format then *.csv
        cc = OpenCC('s2t')
        list_of_files[0] = cc.convert(list_of_files[0])
    elif mode == "ext":
        list_of_files = glob.glob('./result/inference/infer_ext*.candidate') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file) as fp:
        return "\n".join(fp.readline()[:-1].replace(' ', '').split("<q>")) # summary

if __name__ == "__main__":
    origin_text = """（中央社記者張茗喧台北20日電）中央流行疫情指揮中心今天宣布新增3例武漢肺炎境外移入病例，2例為台籍女機師（案760）同事，其中1人有症狀且機上無防護，血清抗體也是陽性，確認此為航空器造成的感染事件。

指揮中心18日公布一名美國境外移入病例，是在國籍航空任職的台籍女機師，由於機師發病前14天曾有台灣、航空器及美國的旅遊史，都不能排除。

據疫調，與案760同行前往美國的同事中，1人於航程途中曾出現咳嗽症狀且未佩戴口罩，其餘同行者皆有佩戴口罩，共匡列59名接觸者須採檢，包括14名家人、4名朋友、26名同事以及15名機師和後艙組員。

根據疾管署新聞稿，中央流行疫情指揮中心今（20）日公布國內新增3例COVID-19確定病例，2名為12月18日公布之案760同事（案765、766），1名為自印尼入境船員（案767）。

指揮中心指出，案765為60多歲紐西蘭籍男性，曾於11月29日飛往美國、12月4日返台，12月12日與案760同班機前往美國，於機上有咳嗽症狀，12月15日返台後進行居家檢疫，12月18日安排接觸者採檢，於今日確診，血清抗體亦為陽性。衛生單位已匡列接觸者8人並安排採檢，３人檢驗結果陰性，其餘檢驗中。

案766為20多歲日本籍男性，曾於12月5日飛往美國，12月7日返台，12月12日案760同班機前往美國，12月15日返台後進行居家檢疫，12月18日安排接觸者採檢，自述12月17日有咳嗽症狀，12月19日出現輕微腹瀉，於今日確診，血清抗體陰性。衛生單位已匡列接觸者21人，將進一步安排採檢及疫調。

指揮中心表示，案760目前接觸者已匡列至60人，已採檢56人，其中2名陽性（案765、766），17名陰性，其餘送驗中。由於案760、765、766曾於12月12日同航班工作，航程均在密閉空間，時間長且有部分時間為無防護接觸（喝水、進食等），而由於案765於機上已有症狀，研判案760、766在機上受案765感染，為一起航空器感染事件。

案767為40多歲印尼籍男性，12月3日來台工作，持有登機前3日內核酸檢驗陰性報告，入境後至防疫旅館檢疫，迄今無症狀。12月18日檢疫期滿後由船務公司安排自費採檢，於今日確診。衛生單位已掌握接觸者共9人，8人為同船船員，列居家隔離，1人為採檢專車司機，因全程有適當防護裝備，列自主健康管理。

指揮中心統計，截至目前國內累計119,405例新型冠狀病毒肺炎相關通報（含117,361例排除），其中766例確診，分別為672例境外移入，55例本土病例，36例敦睦艦隊、2例航空器感染及1例不明；另1例（案530）移除為空號。確診個案中7人死亡，627人解除隔離，132人住院隔離中。（編輯：管中維）1091220
"""
    # with Pool(2) as pool:
    #     pool_output = pool.starmap(get_summary, [("abs", "trans", origin_text), ("ext", "trans", origin_text)])
    # print(json.dumps({"abs":pool_output[0], "ext":pool_output[1]}, ensure_ascii=False))
    # print(pool_output)
    print(get_summary("ext", "trans", origin_text))
    timestamp("inference done")
pass
