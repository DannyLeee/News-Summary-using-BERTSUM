#%%
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
from transformers.activations import ACT2FN
import random

config = BertConfig.from_pretrained('hfl/chinese-bert-wwm', output_hidden_states=True, return_dict=True)
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')

#%%
"""from huggingface source code"""
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
"""------"""

# class Bert(nn.Module):
#     def __init__(self):
#         super(Bert, self).__init__()
#         self.model = BertModel.from_pretrained('hfl/chinese-bert-wwm', config=config)

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         return output

BERT = BertModel.from_pretrained('hfl/chinese-bert-wwm', config=config)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#%%
class Seq2Seq(nn.Module):
    def __init__(self, mode):
        super(Seq2Seq, self).__init__()
        assert mode in ['G', 'R']
        # 把 bert 拿進來
        self.mode = mode
        self.encoder = BERT # encoder
        self.decoder = nn.LSTM(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.hidden2voc = BertLMPredictionHead(self.encoder.config) # output vocabulary size (word distrubution)
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        # encode
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        top_vec = encoder_out.last_hidden_state # (batch_size * seq_len * hidden_size)
        cls_vec = encoder_out.pooler_output # (batch_size * hidden_size)

        # decode
        sentence_list = []

        decoder_out, (h_1, c_1) = self.decoder(cls_vec.unsqueeze(1))
        word = "[CLS]"
        input = h_1
        i = 0
        sentence_list.append(word)

        # print("decoder_out", decoder_out.shape, decoder_out)
        # print("h_1", h_1.shape, h_1)

        while (word != "[SEP]" and ((self.mode=='G' and i < 50) or (self.mode=='R' and i < 511))):
            i += 1
            decoder_out, (h_t, c_t) = self.decoder(input) # output (batch, seq_len, output_size), h (num_layers*num_directions, batch, hidden_size)
            # print("decoder_out", decoder_out.shape)
            # print("h_t", h_t.shape)
            # print("input shape", input.shape)
            # print("----\n")
            input = torch.cat((input, h_t), 1)
            word_distrubution = self.hidden2voc(h_t)
            # print("word dis shape", word_distrubution.shape)

            # sampling: sample a word to be next input
            y_s = random.choices(range(len(word_distrubution)), word_distrubution)
            word = tokenizer.convert_ids_to_tokens(y_s)
            # y_s = torch.argmax(word_distrubution)
            # # print("y_s id ",y_s)
            # word = tokenizer.convert_ids_to_tokens(y_s.item())

            sentence_list.append(word)
        # print(sentence_list)
        return dotdict(cls=cls_vec, top_vec=top_vec, \
            word_dis=word_distrubution, sentence=sentence_list)

class Discriminator(nn.Module): ### 可以用 for sequence classifier ??
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cls = nn.Linear(BERT.config.hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        isReal = self.cls(hidden_states)
        isReal = self.sigmoid(isReal)
        return isReal

# class Reconstructor(nn.Module):
#     def __init__(self):
#         super(Reconstructor, self).__init__()
#         self.bert = Bert(config)

#     def forward(self):
#         return