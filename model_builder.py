#%%
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer

#%%
class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h)
        return sent_scores


#%%
class Summarizer (nn.Module):
    def __init__(self, encoder = "classifier"):
        super(Summarizer, self).__init__()
        # 把 bert 拿進來
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm', output_hidden_states=True)
        # 串接不同的下游
        if (encoder == 'classifier'):
            self.encoder = Classifier(self.bert.config.hidden_size)
        # elif(encoder=='transformer'):
        #     self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
        #                                            args.dropout, args.inter_layers)

    def forward(self, x, segs, clss, attention_mask):
        top_vec = self.bert(input_ids=x, token_type_ids=segs, attention_mask=attention_mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec).squeeze(-1)
        return sent_scores