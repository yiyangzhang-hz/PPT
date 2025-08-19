import json
import os
import random

import numpy as np
import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import IdTok, SplitTok, BertTok, EntTok, BaseTok, NumberTok
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from transformers import LlamaTokenizer
from transformers import AutoTokenizer


class GloveTok(BaseTok):
    return_list = True

    def __init__(self, name: str, path: str):
        super().__init__(name)
        self.vocab = Vocab('english').load(path, as_path=True)

    def t(self, obj: str):
        ids = []
        objs = word_tokenize(str(obj).lower())
        for o in objs:
            if o in self.vocab.obj2index:
                ids.append(self.vocab.obj2index[o])
        return ids or [self.vocab.obj2index[',']]

class LlamaTok(BaseTok):
    return_list = True

    def __init__(self, name, vocab_dir):
        super(LlamaTok, self).__init__(name=name)
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_dir)
        vocab = [self.tokenizer.convert_ids_to_tokens(i) for i in range(self.tokenizer.vocab_size)]
        self.vocab.extend(vocab)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids

class LlamaTok3(BaseTok):
    return_list = True

    def __init__(self, name, vocab_dir):
        super(LlamaTok3, self).__init__(name=name)
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
        vocab = [self.tokenizer.convert_ids_to_tokens(i) for i in range(self.tokenizer.vocab_size)]
        self.vocab.extend(vocab)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids

txt_tok = LlamaTok(name='llama', vocab_dir='D:/不可思议/新闻推荐/模型/LLM/yahma/llama-7b-hf')

print(f"Finished")