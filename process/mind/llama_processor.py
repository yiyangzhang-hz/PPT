import os

import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import IdTok, EntTok, BaseTok
from transformers import LlamaTokenizer


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


class ClickTok(BaseTok):
    def __init__(self, name: str):
        super().__init__(name)
        self.vocab.append(0)
        self.vocab.append(1)
        self.vocab.deny_edit()

    def t(self, obj):
        return int(obj)


class Processor:
    def __init__(self, data_dir, store_dir, nid_vocab_path: str):
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.v2 = True

        os.makedirs(self.store_dir, exist_ok=True)

        self.nid = Vocab(name='nid').load(nid_vocab_path)

    def read_news_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subcat', 'title', 'abs', 'tit_ent', 'abs_ent'],
            usecols=['nid', 'cat', 'subcat', 'title', 'abs'],
        )

    def get_news_tok(self, max_title_len=0, max_abs_len=0):
        txt_tok = LlamaTok(name='llama', vocab_dir='D:/不可思议/新闻推荐/模型/LLM/yahma/llama-7b-hf')

        return UniTok().add_col(Column(
            name='nid',
            tok=IdTok(name='nid', vocab=self.nid)
        )).add_col(Column(
            name='cat',
            tok=txt_tok,
        )).add_col(Column(
            name='subcat',
            tok=txt_tok,
        )).add_col(Column(
            name='title',
            tok=txt_tok,
            max_length=max_title_len,
        )).add_col(Column(
            name='abs',
            tok=txt_tok,
            max_length=max_abs_len,
        ))

    def combine_news_data(self):
        news_train_df = self.read_news_data('MINDtiny_train')
        news_dev_df = self.read_news_data('MINDtiny_dev')
        news_df = pd.concat([news_train_df, news_dev_df])
        news_df = news_df.drop_duplicates(['nid'])
        return news_df

    def analyse_news(self):
        tok = self.get_news_tok(
            max_title_len=0,
            max_abs_len=0
        )
        df = self.combine_news_data()
        tok.read_file(df).analyse()

    def tokenize(self):
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_abs_len=50
        ) #UniTok类别，包含了nid和llama的vocabulary
        news_df = self.combine_news_data() #合并train和dev的news.tsv，包含nid, category, subcategory, title, abstract
        news_tok.read_file(news_df).tokenize().store_data(os.path.join(self.store_dir, 'news-llama'))


if __name__ == '__main__':
    p = Processor(
        data_dir='D:/不可思议/新闻推荐/数据集/MIND/MINDtiny',
        store_dir='D:/不可思议/新闻推荐/代码/文献的开源代码/Legommenders-master/data_mine/MIND-tiny',
        nid_vocab_path='D:/不可思议/新闻推荐/代码/文献的开源代码/Legommenders-master/data_mine/MIND-tiny/news',
    )

    p.tokenize()

    # p.analyse_news()

    # p = Processor(
    #     data_dir='/data1/qijiong/Data/MIND-large/',
    #     store_dir='../../data/MIND-large',
    #     nid_vocab_path='../../data/MIND-large/news',
    # )
    #
    # p.tokenize()
