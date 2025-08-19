from typing import Type

import torch
from pigmento import pnt
from torch import nn

from loader.meta import Meta
from loader.status import Status
from model.common.base_module import BaseModule
from model.common.mediator import Mediator
from model.common.user_plugin import UserPlugin
from model.operators.base_llm_operator import BaseLLMOperator
from model.operators.base_operator import BaseOperator
from model.predictors.base_predictor import BasePredictor
from loader.cacher.repr_cacher import ReprCacher
from loader.column_map import ColumnMap
from loader.embedding.embedding_hub import EmbeddingHub
from loader.data_hub import DataHub
from utils.function import combine_config
from utils.shaper import Shaper
import os
import numpy as np
import itertools


class LegommenderMeta:
    def __init__(
            self,
            item_encoder_class: Type[BaseOperator],
            user_encoder_class: Type[BaseOperator],
            predictor_class: Type[BasePredictor],
    ):
        self.item_encoder_class = item_encoder_class
        self.user_encoder_class = user_encoder_class
        self.predictor_class = predictor_class


class LegommenderConfig:
    def __init__(
            self,
            hidden_size,
            user_config,
            use_neg_sampling: bool = True,
            neg_count: int = 4,
            embed_hidden_size=None,
            item_config=None,
            predictor_config=None,
            use_item_content: bool = True,
            max_item_content_batch_size: int = 0,
            same_dim_transform: bool = True,
            page_size: int = 512,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.item_config = item_config
        self.user_config = user_config
        self.predictor_config = predictor_config or {}

        self.use_neg_sampling = use_neg_sampling
        self.neg_count = neg_count
        self.use_item_content = use_item_content
        self.embed_hidden_size = embed_hidden_size or hidden_size

        self.max_item_content_batch_size = max_item_content_batch_size
        self.same_dim_transform = same_dim_transform

        self.page_size = page_size

        if self.use_item_content:
            if not self.item_config:
                self.item_config = {}
                # raise ValueError('item_config is required when use_item_content is True')
                pnt('automatically set item_config to an empty dict, as use_item_content is True')


class Legommender(BaseModule):
    def __init__(
            self,
            meta: LegommenderMeta, # itme encoder, user encoder, predictor
            status: Status, # 训练，验证，测试
            config: LegommenderConfig, # 一些固定参数，不包括news.tsv和behaviors.tsv的路径
            column_map: ColumnMap, # 对应表，感觉没啥用
            embedding_manager: EmbeddingHub, # 管理embedding的，但是和news.tsv和behaviors.tsv暂时也没啥关系
            user_hub: DataHub, # 包括了behavior.tsv的信息，只有train set的
            item_hub: DataHub, # 包括了news.tsv的信息，而且涵盖了train set和dev set
            user_plugin: UserPlugin = None,
            iftd=False,
    ):
        super().__init__()

        """initializing basic attributes"""
        self.meta = meta #item_encoder, user_encoder, predictor的class
        self.status = status #控制train, dev, test
        self.item_encoder_class  = meta.item_encoder_class
        self.user_encoder_class = meta.user_encoder_class
        self.predictor_class = meta.predictor_class

        self.use_neg_sampling = config.use_neg_sampling
        self.neg_count = config.neg_count

        self.config = config  # type: LegommenderConfig

        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table() #包含了所有cat和English的嵌入字典

        self.user_hub = user_hub
        self.item_hub = item_hub

        self.column_map = column_map  # type: ColumnMap
        self.user_col = column_map.user_col # 'uid'
        self.clicks_col = column_map.clicks_col # 'history'
        self.candidate_col = column_map.candidate_col # 'nid'
        self.label_col = column_map.label_col # 'click'
        self.clicks_mask_col = column_map.clicks_mask_col # '__clicks_mask__'

        """initializing core components"""
        self.flatten_mode = self.user_encoder_class.flatten_mode # False
        self.user_encoder = self.prepare_user_module() # AdaOperator
        self.item_encoder = None
        if self.config.use_item_content:
            self.item_encoder = self.prepare_item_module() # BertOperator
            if iftd:
                item_config = self.item_encoder_class.config_class(**combine_config(
                    config=self.config.item_config,
                    hidden_size=self.config.hidden_size,
                    embed_hidden_size=self.config.embed_hidden_size,
                    input_dim=self.config.embed_hidden_size,
                ))  # 获得有关item的全部所需参数

                # item_config.weights_dir = 'data_mine/MIND-small_sdtd-Llama/llama-7b-split'
                item_config.weights_dir = 'data_mine/MIND-small_sdtd-Bert/bert-12l-split'
                # item_config.weights_dir = 'data_mine/MIND-small_td_NOR-Llama/llama-7b-split'

                self.item_encoder.hidden_weights_eng = torch.from_numpy(np.load(os.path.join(item_config.weights_dir, f'layer_{item_config.layer_split}.npy'))).to(Meta.device)
                self.item_encoder.attention_mask_eng = torch.from_numpy(np.load(os.path.join(item_config.weights_dir, 'mask.npy'))).to(Meta.device)
                self.item_encoder.hidden_weights_eng = self.item_encoder.hidden_weights_eng.view(*self.item_encoder.attention_mask_eng.shape[:2], self.item_encoder.hidden_weights_eng.shape[-1])
        self.item_encoder.ifeng = False

        self.predictor = self.prepare_predictor() # DotPredictor
        self.mediator = Mediator(self) # 要注意这里包含了item和user的信息的，即news.tsv和behavior.tsv

        """initializing extra components"""
        self.user_plugin = user_plugin

        """initializing utils"""
        # special cases for llama
        self.llm_skip = False
        if self.config.use_item_content:
            if isinstance(self.item_encoder, BaseLLMOperator):
                if self.item_encoder.config.layer_split:
                    self.llm_skip = True
                    pnt("LLM SKIP")

        self.shaper = Shaper()
        self.cacher = ReprCacher(self)

        self.loss_func = nn.CrossEntropyLoss() if self.use_neg_sampling else nn.BCEWithLogitsLoss()

    @staticmethod
    def get_sample_size(item_content):
        if isinstance(item_content, torch.Tensor):
            return item_content.shape[0]
        assert isinstance(item_content, dict)
        key = list(item_content.keys())[0]
        return item_content[key].shape[0]

    def get_item_content(self, batch, col):
        if self.cacher.item.cached:
            indices = batch[col]
            shape = indices.shape
            indices = indices.reshape(-1)
            item_repr = self.cacher.item.repr[indices]
            item_repr = item_repr.reshape(*shape, -1)
            # return self.cacher.item.repr[batch[col]]
            return item_repr

        if not self.llm_skip:
            _shape = None
            item_content = self.shaper.transform(batch[col]) #将所有1+K的数据的新闻拼在一块，即title是((1+K) * 数据条数) * 每篇新闻的最大长度(20)，cat是((1+K) * 数据条数) * 1  # batch_size, click_size, max_seq_len
            attention_mask = self.item_encoder.inputer.get_mask(item_content)
            item_content = self.item_encoder.inputer.get_embeddings(item_content)
        else:
            _shape = batch[col].shape
            item_content = batch[col].reshape(-1)
            attention_mask = None

        sample_size = self.get_sample_size(item_content)
        allow_batch_size = self.config.max_item_content_batch_size or sample_size
        batch_num = (sample_size + allow_batch_size - 1) // allow_batch_size

        # item_contents = torch.zeros(sample_size, self.config.hidden_size, dtype=torch.float).to(Setting.device)
        item_contents = self.item_encoder.get_full_placeholder(sample_size).to(Meta.device) #过完整个LLM后的，原来一篇新闻应当是(标题长度)*(embed_hidden_size)的，现在经过linear先映射到低维的hidden_size后再经attention fusion得到一个hidden_size维的向量
        for i in range(batch_num):
            start = i * allow_batch_size
            end = min((i + 1) * allow_batch_size, sample_size)
            mask = None if attention_mask is None else attention_mask[start:end]
            content = self.item_encoder(item_content[start:end], mask=mask)
            item_contents[start:end] = content

        if not self.llm_skip:
            item_contents = self.shaper.recover(item_contents)
        else:
            item_contents = item_contents.view(*_shape, -1) #转换成(数据条数)*(1+k)*(hidden_size)的格式
        return item_contents

    def get_user_content(self, batch):
        if self.cacher.user.cached:
            return self.cacher.user.repr[batch[self.user_col]]

        if self.config.use_item_content and not self.flatten_mode:
            clicks = self.get_item_content(batch, self.clicks_col) #获得每条数据的用户的30条历史点击新闻的hidden_size维向量
        else:
            clicks = self.user_encoder.inputer.get_embeddings(batch[self.clicks_col])
        # 通过对
        user_embedding = self.user_encoder(
            clicks,
            mask=batch[self.clicks_mask_col].to(Meta.device),
        )
        user_embedding = self.fuse_user_plugin(batch, user_embedding)
        return user_embedding

    def fuse_user_plugin(self, batch, user_embedding):
        if self.user_plugin:
            return self.user_plugin(batch[self.user_col], user_embedding)
        return user_embedding

    def forward(self, batch, contrastive=False, rank=False):
        if not contrastive: # 没用对比学习，就是普通的交叉熵
            self.item_encoder.ifeng = False
            if isinstance(batch[self.candidate_col], torch.Tensor) and batch[self.candidate_col].dim() == 1:
                batch[self.candidate_col] = batch[self.candidate_col].unsqueeze(1) # self.candidate_col = 'nid'
            # 一条数据包含1+K个candidates，他们过完LLM的剩余层以及用于映射到传统模型低维度的线性层，最后得到(数据条数)*(1+K)*(hidden_size)的news embedding
            if self.config.use_item_content:
                item_embeddings = self.get_item_content(batch, self.candidate_col) #数据条数 * 1+K * hidden_size
            else:
                item_embeddings = self.embedding_manager(self.clicks_col)(batch[self.candidate_col].to(Meta.device))

            # 获得(数据条数)*(hidden_size)的user embeddding
            user_embeddings = self.get_user_content(batch)

            if self.use_neg_sampling: #如果负采样了
                scores = self.predict_for_neg_sampling(item_embeddings, user_embeddings) #(数据条数*(1+K))的分数，表示每个user对每个item的喜爱程度
                labels = torch.zeros(scores.shape[0], dtype=torch.long, device=Meta.device) #(数据条数,)的tensor，全为0，因为在构造的1+K个candidates中，index为0的表示正样本，其余的均为负样本，也即imp都是1 0 0 0 0的形式
            else:
                scores = self.predict_for_ranking(item_embeddings, user_embeddings)
                labels = batch[self.label_col].float().to(Meta.device)

            if self.status.is_testing or (self.status.is_evaluating and not Meta.simple_dev): #如果在测试
                return scores

            return self.loss_func(scores, labels) #如果有负采样的话就是CrossEntropyLoss

        else:
            if not rank: # 用了对比学习的损失函数
                if isinstance(batch[self.candidate_col], torch.Tensor) and batch[self.candidate_col].dim() == 1:
                    batch[self.candidate_col] = batch[self.candidate_col].unsqueeze(1) # self.candidate_col = 'nid'

                self.item_encoder.ifeng = False
                item_embeddings = self.get_item_content(batch, self.candidate_col) #数据条数 * 1+K * hidden_size

                self.item_encoder.ifeng = True
                item_embeddings_eng = self.get_item_content(batch, self.candidate_col)  # 数据条数 * 1+K * hidden_size

                def nt_xent_loss(embedding_anchor, embedding_positive, temperature=0.05):
                    # 将batch_size和1+K个samples展开
                    batch_size, num, dim = embedding_anchor.size()
                    embedding_anchor = embedding_anchor.view(-1, dim)  # (batch * num, dim)
                    embedding_anchor = embedding_anchor.view(-1, dim)  # (batch * num, dim)
                    embedding_positive = embedding_positive.view(-1, dim)  # (batch * num, dim)
                    embedding_positive = embedding_positive.view(-1, dim)  # (batch * num, dim)
                    embedding_anchor = torch.nn.functional.normalize(embedding_anchor, p=2, dim=1)  # 每行归一化为单位范数
                    embedding_positive = torch.nn.functional.normalize(embedding_positive, p=2, dim=1)

                    # 合并嵌入
                    embeddings = torch.cat([embedding_anchor, embedding_positive], dim=0)  # 合并 anchor 和 positive

                    # 计算相似度矩阵
                    similarity_matrix = torch.mm(embeddings, embeddings.t())  # (2N, 2N)

                    # 去掉自相似度
                    batch_size = embedding_anchor.size(0)
                    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
                    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))  # 忽略对角线

                    # 构造目标
                    half = embeddings.size(0) // 2
                    labels_chinese = torch.arange(half, device=similarity_matrix.device) + half  # 对于中文 (0..half-1)
                    labels_english = torch.arange(half, device=similarity_matrix.device)  # 对于英文 (half..n-1)
                    labels = torch.cat([labels_chinese, labels_english], dim=0)

                    # 计算对比损失
                    loss = torch.nn.functional.cross_entropy(similarity_matrix / temperature, labels)
                    return loss

                def nt_xent_loss2(embedding_anchor, embedding_positive, temperature=0.07):
                    # 将batch_size和1+K个samples展开
                    batch_size, num, dim = embedding_anchor.size()
                    embedding_anchor = embedding_anchor.view(-1, dim)  # (batch * num, dim)
                    embedding_anchor = embedding_anchor.view(-1, dim)  # (batch * num, dim)
                    embedding_positive = embedding_positive.view(-1, dim)  # (batch * num, dim)
                    embedding_positive = embedding_positive.view(-1, dim)  # (batch * num, dim)
                    embedding_anchor = torch.nn.functional.normalize(embedding_anchor, p=2, dim=1)  # 每行归一化为单位范数
                    embedding_positive = torch.nn.functional.normalize(embedding_positive, p=2, dim=1)

                    # 合并嵌入
                    embeddings = torch.cat([embedding_anchor, embedding_positive], dim=0)  # 合并 anchor 和 positive

                    # 计算相似度矩阵
                    similarity_matrix = torch.mm(embeddings, embeddings.t())  # (2N, 2N)

                    # 去掉自相似度
                    batch_size = embedding_anchor.size(0)
                    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
                    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))  # 忽略对角线

                    # 只关注同一篇新闻的英文中文嵌入尽可能接近，同时不同新闻的英文嵌入尽可能远离
                    similarity_matrix_ENG2ENG = similarity_matrix[0:(batch_size), 0:(batch_size)]
                    similarity_matrix_ENG2LIN = similarity_matrix[0:(batch_size),(batch_size):(2 * batch_size)]
                    diagonal_elements = similarity_matrix_ENG2LIN.diagonal()
                    similarity_matrix = torch.cat((similarity_matrix_ENG2ENG, diagonal_elements.unsqueeze(1)), dim=1)

                    # 构造目标
                    labels = torch.full((160,), 160, device=similarity_matrix.device)

                    # 计算对比损失
                    loss = torch.nn.functional.cross_entropy(similarity_matrix / temperature, labels)
                    return loss

                def my_mse_loss(english_embeddings, lingual_embeddings):
                    batch_size, num, dim = english_embeddings.size()
                    english_embeddings = english_embeddings.view(-1, dim)  # (batch * num, dim)
                    english_embeddings = english_embeddings.view(-1, dim)  # (batch * num, dim)
                    lingual_embeddings = lingual_embeddings.view(-1, dim)  # (batch * num, dim)
                    lingual_embeddings = lingual_embeddings.view(-1, dim)  # (batch * num, dim)

                    # Ensure embeddings are normalized to unit norm (optional, if required)
                    # english_embeddings = torch.nn.functional.normalize(english_embeddings, p=2, dim=1)
                    # lingual_embeddings = torch.nn.functional.normalize(lingual_embeddings, p=2, dim=1)

                    # Compute MSE Loss
                    loss = 10000 * torch.nn.functional.mse_loss(english_embeddings, lingual_embeddings)
                    return loss
                return nt_xent_loss(item_embeddings_eng, item_embeddings)

            else: # 用排序损失
                if isinstance(batch[self.candidate_col], torch.Tensor) and batch[self.candidate_col].dim() == 1:
                    batch[self.candidate_col] = batch[self.candidate_col].unsqueeze(1) # self.candidate_col = 'nid'

                self.item_encoder.ifeng = False
                item_embeddings = self.get_item_content(batch, self.candidate_col) #数据条数 * 1+K * hidden_size
                user_embeddings = self.get_user_content(batch)
                scores1 = self.predict_for_neg_sampling(item_embeddings, user_embeddings)  # (数据条数*(1+K))的分数，表示每个user对每个item的喜爱程度

                self.item_encoder.ifeng = True
                item_embeddings_eng = self.get_item_content(batch, self.candidate_col)  # 数据条数 * 1+K * hidden_size
                user_embeddings_eng = self.get_user_content(batch)
                scores2 = self.predict_for_neg_sampling(item_embeddings_eng, user_embeddings_eng)  # (数据条数*(1+K))的分数，表示每个user对每个item的喜爱程度

                def calculate_pairwise_differences(scores):
                    # 生成所有可能的两两组合索引对
                    indices = list(itertools.combinations(range(5), 2))

                    # 初始化结果列表
                    result = []

                    # 对每个batch中的序列计算差值
                    for seq in scores:
                        differences = [seq[j] - seq[i] for i, j in indices]
                        result.append(differences)

                    # 转换为Tensor，返回一个形状为 (batch_size, 10) 的结果
                    return torch.tensor(result)

                def nt_xent_loss(embedding_anchor, embedding_positive, temperature=0.05):
                    # 将batch_size和1+K个samples展开
                    embedding_anchor = embedding_anchor.unsqueeze(1)
                    embedding_positive = embedding_positive.unsqueeze(1)
                    batch_size, num, dim = embedding_anchor.size()
                    embedding_anchor = embedding_anchor.view(-1, dim)  # (batch * num, dim)
                    embedding_anchor = embedding_anchor.view(-1, dim)  # (batch * num, dim)
                    embedding_positive = embedding_positive.view(-1, dim)  # (batch * num, dim)
                    embedding_positive = embedding_positive.view(-1, dim)  # (batch * num, dim)
                    embedding_anchor = torch.nn.functional.normalize(embedding_anchor, p=2, dim=1)  # 每行归一化为单位范数
                    embedding_positive = torch.nn.functional.normalize(embedding_positive, p=2, dim=1)

                    # 合并嵌入
                    embeddings = torch.cat([embedding_anchor, embedding_positive], dim=0)  # 合并 anchor 和 positive

                    # 计算相似度矩阵
                    similarity_matrix = torch.mm(embeddings, embeddings.t())  # (2N, 2N)

                    # 去掉自相似度
                    batch_size = embedding_anchor.size(0)
                    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
                    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))  # 忽略对角线

                    # 构造目标
                    half = embeddings.size(0) // 2
                    labels_chinese = torch.arange(half, device=similarity_matrix.device) + half  # 对于中文 (0..half-1)
                    labels_english = torch.arange(half, device=similarity_matrix.device)  # 对于英文 (half..n-1)
                    labels = torch.cat([labels_chinese, labels_english], dim=0)

                    # 计算对比损失
                    loss = torch.nn.functional.cross_entropy(similarity_matrix / temperature, labels)
                    return loss

                scores1_diff = calculate_pairwise_differences(scores1)
                scores2_diff = calculate_pairwise_differences(scores2)

                mse_loss_fn = nn.MSELoss()
                loss = mse_loss_fn(scores1_diff,scores2_diff)

            # return nt_xent_loss(user_embeddings,user_embeddings_eng)
            return loss

    def predict_for_neg_sampling(self, item_embeddings, user_embeddings):
        batch_size, candidate_size, hidden_size = item_embeddings.shape #获取相关信息，batch_size=数据条数，candidate_size=1+K，hidden_size就是传统模型中的embedding维度
        if self.predictor.keep_input_dim:
            return self.predictor(user_embeddings, item_embeddings)
        # if user_embeddings: B, S, D
        # user_embeddings = user_embeddings.unsqueeze(1).repeat(1, candidate_size, 1)  # B, K+1, D
        # user_embeddings = user_embeddings.view(-1, hidden_size)
        user_embeddings = self.user_encoder.prepare_for_predictor(user_embeddings, candidate_size) #扩展为1+K份，并squeeze为(数据条数*(1+K))*(hidden_size)
        item_embeddings = item_embeddings.view(-1, hidden_size) #同样squeeze为(数据条数*(1+K))*(hidden_size)
        scores = self.predictor(user_embeddings, item_embeddings) #(数据条数*(1+K))个点积后的结果
        scores = scores.view(batch_size, -1) #再unqueeze为数据条数*(1+K)，其中的一条数据表示user对这1+K个item的喜爱程度
        return scores

    def predict_for_ranking(self, item_embeddings, user_embeddings):
        item_embeddings = item_embeddings.squeeze(1)
        scores = self.predictor(user_embeddings, item_embeddings)
        return scores

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    def prepare_user_module(self):
        user_config = self.user_encoder_class.config_class(**combine_config(
            config=self.config.user_config,
            hidden_size=self.config.hidden_size,
            embed_hidden_size=self.config.embed_hidden_size,
            input_dim=self.config.hidden_size,
        ))

        if self.flatten_mode:
            user_config.inputer_config['item_hub'] = self.item_hub

        return self.user_encoder_class(
            config=user_config,
            hub=self.user_hub,
            embedding_manager=self.embedding_manager,
            target_user=True,
        )

    def prepare_item_module(self):
        item_config = self.item_encoder_class.config_class(**combine_config(
            config=self.config.item_config,
            hidden_size=self.config.hidden_size,
            embed_hidden_size=self.config.embed_hidden_size,
            input_dim=self.config.embed_hidden_size,
        )) #获得有关item的全部所需参数

        return self.item_encoder_class(
            config=item_config,
            hub=self.item_hub,
            embedding_manager=self.embedding_manager,
            target_user=False,
        )

    def prepare_predictor(self):
        if self.config.use_neg_sampling and not self.predictor_class.allow_matching:
            raise ValueError(f'{self.predictor_class.__name__} does not support negative sampling')

        if not self.config.use_neg_sampling and not self.predictor_class.allow_ranking:
            raise ValueError(f'{self.predictor_class.__name__} only supports negative sampling')

        predictor_config = self.predictor_class.config_class(**combine_config(
            config=self.config.predictor_config,
            hidden_size=self.config.hidden_size,
            embed_hidden_size=self.config.embed_hidden_size,
        ))

        return self.predictor_class(config=predictor_config)
    """"""
    def get_parameters(self):
        pretrained_parameters = []
        other_parameters = []
        pretrained_signals = self.item_encoder.get_pretrained_parameter_names() #transformers

        pretrained_names = []
        other_names = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_pretrained = False
            for pretrained_name in pretrained_signals:
                if name.startswith(f'item_encoder.{pretrained_name}'):
                    pretrained_names.append((name, param.data.shape))
                    pretrained_parameters.append(param)
                    is_pretrained = True
                    break

            if not is_pretrained:
                # pnt(f'[N] {name} {param.data.shape}')
                other_names.append((name, param.data.shape))
                other_parameters.append(param)

        for name, shape in pretrained_names:
            pnt(f'[P] {name} {shape}')
        for name, shape in other_names:
            pnt(f'[N] {name} {shape}')

        return pretrained_parameters, other_parameters