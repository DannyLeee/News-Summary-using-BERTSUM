import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-root_path", type=str, required=True)
parser.add_argument("-file_name", type=str, required=True, help="without extention")
parser.add_argument("-mode", type=str, choices=['3', 's10', 'w10'], default='3')
args = parser.parse_args()

with open("voc.pkl", 'rb') as fp:
    voc = pickle.load(fp)
with open("word_len.pkl", 'rb') as fp:
    word_len = pickle.load(fp)
with open("sentence_len.pkl", 'rb') as fp:
    sentence_len = pickle.load(fp)

predict = []
# FILE_PATH = "./result/classifier_step180_word10.candidate"
# FILE_PATH = "./result/transformer_step180_word10.candidate"
# FILE_PATH = "./result/baseline_step180_word10.candidate"
# FILE_PATH = "./result/raw_lead3_word10.txt"

# input raw prediction
print("reading raw prediction ...")
with open(f"{args.root_path + args.file_name}.candidate", "r") as fileIn:
    line = fileIn.readline()
    l = 0
    while (line != ""):
        sentence_list = line.split("<q>")
        sentence = ""
        for _, s in enumerate(sentence_list):
            sentence += s
            if args.mode == "w10" and len(sentence) >= word_len[l]*0.1:
                break
            elif args.mode == "s10" and _ >= sentence_len[l]*0.1:
                break
        predict.append(sentence)
        line = fileIn.readline()
        l += 1

# convert raw prediction to id
print("convert to id ...")
bert_dict = {'input_ids': []}
for p in predict:
    ls = []
    for w in p:
        try:
            ls += [voc[w]]
        except:
            if w != '\n':
                print(w, "OOV")
    bert_dict['input_ids'] += [ls]

print("writing output file ...")
args.file_name += "" if args.mode == '3' else '_'+args.mode
with open(f"{args.root_path + args.file_name}.txt", "w") as fileOut:
    for token in bert_dict['input_ids']:
        for i in token:
            if (i != 0):
                fileOut.write(str(i) + ' ')
        fileOut.write('\n')

# evaluation
with open("result/rouge_config.xml", "w") as fp:
    fp.write(f"""<ROUGE-EVAL version="1.55">
    <EVAL ID="1">
        <MODEL-ROOT>{args.root_path}</MODEL-ROOT>
        <PEER-ROOT>{args.root_path}</PEER-ROOT>
        <INPUT-FORMAT TYPE="SPL">
        </INPUT-FORMAT>
        <PEERS>
            <P ID="1">{args.file_name}.txt</P>
        </PEERS>
        <MODELS>
            <M ID="A">gold.txt</M>
        </MODELS>
    </EVAL>
</ROUGE-EVAL>""")

print("evaluation ...")
os.system("/home/danny//.local/lib/python3.6/site-packages/rouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -e /home/danny//.local/lib/python3.6/site-packages/rouge/tools/ROUGE-1.5.5/data -c 95 -m -r 1000 -n 2 -a ./result/rouge_config.xml")