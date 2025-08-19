import warnings

import torch
from oba import Obj
from pigmento import pnt
from torch import nn

from loader.data_hubs import DataHubs
from loader.data_sets import DataSets
from loader.depot.depot_hub import DepotHub
from loader.depots import Depots
from loader.meta import Meta, Phases, DatasetType
from loader.status import Status
from model.common.user_plugin import UserPlugin
from model.inputer.concat_inputer import ConcatInputer
from model.inputer.flatten_seq_inputer import FlattenSeqInputer
from model.inputer.natural_concat_inputer import NaturalConcatInputer
from model.legommender import Legommender, LegommenderConfig, LegommenderMeta
from loader.column_map import ColumnMap
from loader.embedding.embedding_hub import EmbeddingHub
from loader.resampler import Resampler
from loader.data_loader import DataLoader
from loader.data_hub import DataHub
from loader.class_hub import ClassHub


class Controller:
    def __init__(self, data, embed, model, exp, iftd=False):
        self.data = data #初始化data的参数
        self.embed = embed #初始化embed的参数
        self.model = model #初始化model的参数
        self.exp = exp #初始化exp的参数
        self.modes = self.parse_mode() #默认为{'layer', 'split', 'llm', 'test'}

        self.status = Status() #设置train dev test

        if 'MIND' in self.data.name.upper(): #新闻推荐时显然yes
            Meta.data_type = DatasetType.news #就是返回'news'
        else:
            Meta.data_type = DatasetType.book
        pnt('dataset type: ', Meta.data_type) #打印此时所用的数据集，即news，用于新闻推荐

        pnt('build column map ...') #构建列映射
        self.column_map = ColumnMap(**Obj.raw(self.data.user)) #只是用来改列名的一个映射，不过get不到他这样的意义是啥

        # depots and data hubs initialization
        self.depots = Depots(user_data=self.data.user, modes=self.modes, column_map=self.column_map) #初始化source domain的数据仓库
        self.hubs = DataHubs(depots=self.depots) #source domain的behavior信息
        self.item_hub = DataHub( #source domain的news信息
            depot=self.data.item.depot,
            order=self.data.item.order,
            append=self.data.item.append,
        )
        if self.data.item.union:
            for depot in self.data.item.union:
                self.item_hub.depot.union(DepotHub.get(depot))

        # legommender components initialization
        operator_set = ClassHub.operators() #包含了Ada, Attention, BaseLLM, CNN, GRU等一系列基础模块的operator.py
        predictor_set = ClassHub.predictors() #包括了DCN, DeepFM, Dot等一系列基础模块的predictor.py

        self.item_operator_class = None
        if self.model.meta.item: #加载item的operator
            self.item_operator_class = operator_set(self.model.meta.item)
        self.user_operator_class = operator_set(self.model.meta.user) #加载user的operator
        self.predictor_class = predictor_set(self.model.meta.predictor) #加载predictor
        self.legommender_meta = LegommenderMeta(
            item_encoder_class=self.item_operator_class,
            user_encoder_class=self.user_operator_class,
            predictor_class=self.predictor_class,
        ) #包含了item_encoder, user_encoder, predictor

        pnt(f'Selected Item Encoder: {str(self.item_operator_class.__name__) if self.item_operator_class else "null"}')
        pnt(f'Selected User Encoder: {str(self.user_operator_class.__name__)}')
        pnt(f'Selected Predictor: {str(self.predictor_class.__name__)}')
        pnt(f'Use Negative Sampling: {self.model.config.use_neg_sampling}')
        pnt(f'Use Item Content: {self.model.config.use_item_content}')

        self.legommender_config = LegommenderConfig(**Obj.raw(self.model.config)) # 包含各种参数

        # embedding initialization
        skip_cols = [self.column_map.candidate_col] if self.legommender_config.use_item_content else []
        self.embedding_hub = EmbeddingHub(
            hidden_size=self.legommender_config.embed_hidden_size, # 768 for BERT
            same_dim_transform=self.model.config.same_dim_transform, # false
        )
        for embedding_info in self.embed.embeddings:
            self.embedding_hub.load_pretrained_embedding(**Obj.raw(embedding_info)) #加载token的embedding，对于BERT就是(30522, 768)的tensor
        self.embedding_hub.register_depot(self.hubs.a_hub(), skip_cols=skip_cols)
        self.embedding_hub.register_vocab(ConcatInputer.vocab)
        self.embedding_hub.register_vocab(FlattenSeqInputer.vocab)
        if self.model.config.use_item_content: #使用item内容
            self.embedding_hub.register_depot(self.item_hub) #这里第一次产生了self里面关于english的embedding，也就是bert/llama所最初的toekn-embedding对照表
            lm_col = self.data.item.lm_col or 'title'
            if self.embedding_hub.has_col(lm_col):
                self.embedding_hub.clone_vocab(
                    col_name=NaturalConcatInputer.special_col,
                    clone_col_name=self.data.item.lm_col or 'title'
                )
            else:
                warnings.warn(f'cannot find lm column in item depot, please ensure no natural inputer is used')
        cat_embeddings = self.embedding_hub(ConcatInputer.vocab.name)  # type: nn.Embedding
        cat_embeddings.weight.data[ConcatInputer.PAD] = torch.zeros_like(cat_embeddings.weight.data[ConcatInputer.PAD])

        # user plugin initialization
        user_plugin = None
        if self.data.user.plugin: # 默认false
            user_plugin = UserPlugin(
                depot=DepotHub.get(self.data.user.plugin),
                hidden_size=self.model.config.hidden_size,
                select_cols=self.data.user.plugin_cols,
            )

        # legommender initialization
        # self.legommender = self.legommender_class(
        self.legommender = Legommender(
            meta=self.legommender_meta, # item encoder, user encoder, predictor
            status=self.status, # 训练，验证，测试
            config=self.legommender_config, # 一些固定参数，不包括news.tsv和behavior.tsv
            column_map=self.column_map, #表头的对照表
            embedding_manager=self.embedding_hub, # 管理embedding的，但是和news.tsv和behavior.tsv也没啥关系
            user_hub=self.hubs.a_hub(), # 包括了behavior.tsv的信息，只有train set的
            item_hub=self.item_hub, # 包括了news.tsv的信息，而且涵盖了train set和dev set
            user_plugin=user_plugin, # None
            iftd=iftd, #判断是否是target数据集的legommender
        ) # 包括了loss_func，如果负采样的话就是CrossEntropy
        self.resampler = Resampler(
            legommender=self.legommender,
            item_hub=self.item_hub, # 包括了news.tsv的信息，而且涵盖了train set和dev set
            status=self.status, # 训练，验证，测试
        )

        if self.legommender_config.use_neg_sampling:
            self.depots.negative_filter(self.column_map.label_col) # 负采样处理，每个imp中抽取-1和4个-0的组成一组

        if self.exp.policy.use_cache:
            for depot in self.depots.depots.values():
                depot.start_caching()

        # data sets initialization
        self.sets = DataSets(hubs=self.hubs, resampler=self.resampler) #数据集

    def parse_mode(self):
        modes = set(self.exp.mode.lower().split('_'))
        if Phases.train in modes: #如果是train
            modes.add(Phases.dev) #需要额外添加dev，反之如果test的话就不用添加了
        return modes

    def get_loader(self, phase):
        return DataLoader(
            resampler=self.resampler,
            user_set=self.sets.user_set, # 包含训练集和测试集的全部behavior.tsv信息
            dataset=self.sets[phase], # 训练集、测试集，主要是behavior.tsv的信息，
            shuffle=phase == Phases.train,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
            num_workers=5,
            # collate_fn=self.stacker,
        )
