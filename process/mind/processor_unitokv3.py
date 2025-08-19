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

class Processor:
    """初始化，输入data_dir和用于保存的store_dir"""
    def __init__(self, data_dir, store_dir, glove=None, imp_list_path: str = None):
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.glove = glove
        self.imp_list = json.load(open(imp_list_path, 'r')) if imp_list_path else None

        # if os.path.exists(self.store_dir):
        #     c = input(f'{self.store_dir} exists, press Y to continue, or press any other to exit.')
        #     if c.upper() != 'Y':
        #         exit(0)

        os.makedirs(self.store_dir, exist_ok=True) #创建用于保存的路径

        self.train_store_dir = os.path.join(self.store_dir, 'MINDsmall_train') #用于保存训练集
        self.dev_store_dir = os.path.join(self.store_dir, 'MINDsmall_dev') #用于保存测试集

        self.nid = Vocab(name='nid') #Create a news id vocab, commonly used in news data, history data, and interaction data.
        self.uid = Vocab(name='uid') #Create a user id vocab, commonly used in user data and interaction data.

    """读取news.tsv"""
    def read_news_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subcat', 'title', 'url', 'abs', 'tit_ent', 'abs_ent'],
            usecols=['nid', 'cat', 'subcat', 'title', 'abs'], #不考虑entity
        )

    """读取behaviors.tsv，用于用户建模的部分"""
    def read_user_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'behaviors.tsv'),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'],
            usecols=['uid', 'history'] #只读取history
        )

    """读取behaviors.tsv，用于train/test的部分"""
    def _read_inter_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'behaviors.tsv'),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'],
            usecols=['imp', 'uid', 'predict'] #只读取impression和predict
        )

    """提取impressions中的负例"""
    def read_neg_data(self, mode):
        df = self._read_inter_data(mode)
        data = dict(uid=[], neg=[])
        for line in df.itertuples():
            if line.uid in data['uid']:
                continue
            # 如果还没有出现过这个uid
            predicts = line.predict.split(' ')
            negs = []
            for predict in predicts:
                nid, click = predict.split('-')
                if not int(click):
                    negs.append(nid)

            data['uid'].append(line.uid)
            data['neg'].append(' '.join(negs))
        return pd.DataFrame(data)

    """读取交互数据，转换为dataframe格式"""
    def read_inter_data(self, mode) -> pd.DataFrame:
        df = self._read_inter_data(mode)
        data = dict(imp=[], uid=[], nid=[], click=[])
        # imp其实是behaviors.tsv中的ID；uid是userID,nid是impressions中所有的news ID；click是impression中所有的点击还是没点击的0-1标注
        for line in df.itertuples():
            predicts = line.predict.split(' ')
            data['imp'].extend([line.imp] * len(predicts))
            data['uid'].extend([line.uid] * len(predicts))
            for predict in predicts:
                nid, click = predict.split('-')
                data['nid'].append(nid)
                data['click'].append(int(click))
        return pd.DataFrame(data)

    """初始化一个用于tokenizer news.tsv的UniTok"""
    def get_news_tok(self, max_title_len=0, max_abs_len=0):
        if self.glove: #如果存在glove，就用GloveTok来tokenizer
            txt_tok = GloveTok(name='english', path=self.glove)
        else: #如果不存在，默认使用BertTok来tokenizer
            # txt_tok = LlamaTok(name='llama', vocab_dir='D:/不可思议/新闻推荐/模型/LLM/llama-2-7b-hf')
            # txt_tok = LlamaTok3(name='llama', vocab_dir='D:/不可思议/新闻推荐/模型/LLM/llama3_hf/llama-3-8b')
            txt_tok = BertTok(name='bert', vocab_dir='bert-base-uncased')
        # 创建一个news UniTok object，然后往里面添加列
        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.nid)
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

    """初始化一个用于tokenizer behaviors.tsv的UniTok，其中包含了'uid': ['uid'], 'nid': ['history']"""
    def get_user_tok(self, max_history: int = 0):
        user_ut = UniTok() #初始化为一个UniTok
        user_ut.add_col(Column(
            tok=IdTok(vocab=self.uid)
        )).add_col(Column(
            name='history',
            tok=SplitTok(
                sep=' ',
                vocab=self.nid
            ),
            max_length=max_history,
            slice_post=True,
        ))
        return user_ut

    """依次tokenizer behaviors.tsv中的userID和neg，不同的数据采用不同的tokenizer方式"""
    def get_neg_tok(self, max_neg: int = 0):
        neg_ut = UniTok()
        neg_ut.add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            name='neg',
            tok=SplitTok(
                sep=' ',
                vocab=self.nid
            ),
            max_length=max_neg,
            slice_post=True,
        ))
        return neg_ut

    """Create an interaction UniTok object"""
    def get_inter_tok(self):
        return UniTok().add_index_col(
            name='index'
        ).add_col(Column(
            name='imp',
            tok=EntTok,
        )).add_col(Column(
            tok=EntTok(vocab=self.uid)
        )).add_col(Column(
            tok=EntTok(vocab=self.nid)
        )).add_col(Column(
            tok=NumberTok(name='click', vocab_size=2)
        ))

    """拼接train和dev中的news.tsv，并去除重复项"""
    def combine_news_data(self):
        news_train_df = self.read_news_data('MINDsmall_train') #dataframe，包含NewsID, cate, subcate, title, abstract这五列
        news_dev_df = self.read_news_data('MINDsmall_dev') #dataframe，包含NewsID, cate, subcate, title, abstract这五列
        news_df = pd.concat([news_train_df, news_dev_df])
        news_df = news_df.drop_duplicates(['nid'])
        return news_df

    """拼接train和dev中的behaviors.tsv，但是去除重复项应该是不合理的，因为同一个userID可以有多个不同时刻的impressions啊"""
    def combine_user_df(self):
        user_train_df = self.read_user_data('MINDsmall_train')
        user_dev_df = self.read_user_data('MINDsmall_dev')

        user_df = pd.concat([user_train_df, user_dev_df])
        user_df = user_df.drop_duplicates(['uid'])
        return user_df

    """拼接train和dev中的negs"""
    def combine_neg_df(self):
        neg_train_df = self.read_neg_data('MINDsmall_train')
        neg_dev_df = self.read_neg_data('MINDsmall_dev')

        neg_df = pd.concat([neg_train_df, neg_dev_df])
        neg_df = neg_df.drop_duplicates(['uid'])
        return neg_df

    """拼接train和dev中的交互数据"""
    def combine_inter_df(self):
        inter_train_df = self.read_inter_data('MINDsmall_train')
        inter_dev_df = self.read_inter_data('MINDsmall_dev')
        inter_dev_df.imp += max(inter_train_df.imp)

        inter_df = pd.concat([inter_train_df, inter_dev_df])
        return inter_df

    """将一个list l按照portions来进行分割"""
    def splitter(self, l: list, portions: list):
        if self.imp_list:
            l = self.imp_list
        else:
            random.shuffle(l)
        json.dump(l, open(os.path.join(self.store_dir, 'imp_list.json'), 'w'))

        portions = np.array(portions)
        portions = portions * 1.0 / portions.sum() * len(l) #需要分割的元素数量
        portions = list(map(int, portions)) #取整
        portions[-1] = len(l) - sum(portions[:-1]) #调整最后一个分割的大小，保证所有部分之和等于原list的长度

        pos = 0 #position，用于跟踪当前的切片位置，确保连续切割
        parts = []
        # 进行分割
        for i in portions:
            parts.append(l[pos: pos+i])
            pos += i
        return parts

    """读取dataframe格式的交互数据，把dev进一步细分为验证集和测试集，五五分成"""
    def reassign_inter_df_v2(self):
        inter_train_df = self.read_inter_data('MINDsmall_train')
        inter_df = self.read_inter_data('MINDsmall_dev')

        imp_list = inter_df.imp.drop_duplicates().to_list()

        dev_imps, test_imps = self.splitter(imp_list, [5, 5]) #对半分
        inter_dev_df, inter_test_df = [], []

        inter_groups = inter_df.groupby('imp')
        for imp, imp_df in inter_groups:
            if imp in dev_imps:
                inter_dev_df.append(imp_df)
            else:
                inter_test_df.append(imp_df)
        return inter_train_df, \
               pd.concat(inter_dev_df, ignore_index=True), \
               pd.concat(inter_test_df, ignore_index=True)

    """"""
    def analyse_news(self):
        tok = self.get_news_tok(
            max_title_len=0,
            max_abs_len=0
        )
        df = self.combine_news_data()
        tok.read(df).analyse()

    """"""
    def analyse_user(self):
        tok = self.get_user_tok(max_history=0)
        df = self.combine_user_df()
        tok.read(df).analyse()

    """"""
    def analyse_inter(self):
        tok = self.get_inter_tok()
        df = self.combine_inter_df()
        tok.read_file(df).analyse()

    """进行已将原来的dev五五分成dev和test的数据集tokenizer的过程"""
    def tokenize(self):
        # 对news进行tokenizer并保存结果
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_abs_len=50
        )
        news_df = self.combine_news_data()
        news_tok.read_file(news_df).tokenize().store_data(os.path.join(self.store_dir, 'news'))

        # 对user (主要是history)进行tokenizer并保存结果
        user_tok = self.get_user_tok(max_history=30)
        user_df = self.combine_user_df()
        user_tok.read(user_df).tokenize().store(os.path.join(self.store_dir, 'user'))

        # 对interaction进行tokenizer并保存结果
        inter_dfs = self.reassign_inter_df_v2()
        for inter_df, mode in zip(inter_dfs, ['train', 'dev', 'test']):
            inter_tok = self.get_inter_tok()
            inter_tok.read_file(inter_df).tokenize().store_data(os.path.join(self.store_dir, mode))

    """对原始的dev进行tokenizer的过程"""
    def tokenize_original_dev(self):
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_abs_len=50
        )
        news_df = self.combine_news_data() #dataframe，包含NewsID, cate, subcate, title, abstract这五列，已经去除了重复项
        # UniTok，包含了news_df的原始data信息，有vocabs，vovcabs中的nid, cat, subcat, english中包含了i2o和o2i的字典
        # cols就是data这个dataframe中的列表头，然后每个Column里面包含了原始数据经o2i的相当于tokenize的结果
        news_tok.read_file(news_df).tokenize()#tokenize和store_data都是UniTok库的unitok.py里面的

        user_tok = self.get_user_tok(max_history=30)
        user_df = self.combine_user_df() #dataframe，包含UserID, history这两列，已经去除了重复项
        # UniTok，包含了user_df的原始data信息，然后有2个vocab_depots，分别是uid, nid
        # 每个vocab_depot包含了从id到obj和从obj到id的VocabMap，uid→uid，nid→history
        user_tok.read(user_df).tokenize() #这个tokenize是UniTok库的unitok.py里面的

        inter_df = self.read_inter_data('MINDsmall_dev') #dataframe，imp, uid, nid, click四列，imp表示imp_id，uid是userID，nid是一篇newsID，click是0/1，是将用户的impression拆成了好多行
        inter_tok = self.get_inter_tok()
        inter_tok.read_file(inter_df).tokenize().store_data(os.path.join(self.store_dir, 'dev-original')) #这个tokenize是UniTok库的unitok.py里面的

    """对负例进行tokenizer的过程"""
    def tokenize_neg(self):
        print('tokenize neg')
        self.uid.load(os.path.join(self.store_dir, 'user')) #加载news的tokenizer的结果
        self.nid.load(os.path.join(self.store_dir, 'news')) #加载user (history)的tokenizer的结果

        print('combine neg df')
        neg_df = self.combine_neg_df() #拼接train和dev中的negs
        print('get neg tok')
        neg_tok = self.get_neg_tok()
        neg_tok.read(neg_df).tokenize().store(os.path.join(self.store_dir, 'neg'))


if __name__ == '__main__':
    language_list = ['SWH', 'SOM', 'sCMN', 'CMN', 'JPN', 'TUR', 'TAM', 'VIE', 'THA', 'RON', 'FIN', 'KAT', 'HAT', 'IND', 'GRN']

    for lingual in language_list:

        p = Processor(
            data_dir=f'D:/不可思议/新闻推荐/数据集/MIND/MINDsmall',
            store_dir=f'D:/不可思议/新闻推荐/代码/文献的开源代码/Lego-title_A40/data_mine/MIND-small/bert',
            # imp_list_path = 'D:/不可思议/新闻推荐/数据集/mind/MIND_cs/data_mine_5/MIND-small_td/llama/imp_list.json'
        )
        p.tokenize()
        p.tokenize_neg()
        p.tokenize_original_dev()
