import datetime
import json
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import pigmento
import torch
from pigmento import pnt
from oba import Obj
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.meta import Meta, Phases
from loader.controller import Controller
# from loader.mode_hub import ModeHub
from model.operators.base_llm_operator import BaseLLMOperator
from utils.config_init import ConfigInit
from utils.function import seeding
from utils.gpu import GPU
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.monitor import Monitor
from loader.pager.llm_split_pager import LLMSplitPager
from utils.structure import Structure
from utils.submission import Submission

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Worker:
    def __init__(self, config):
        self.config = config  # 加载参数，包括embed, model, exp, data以及一堆default参数
        # data来自config/data/mind-llama.yaml
        # model来自config/model/llama-naml.yaml
        # exp来自config/exp/llama-split.yaml
        self.data_sd, self.data_td, self.embed, self.model_sd, self.model_td, self.exp = \
            self.config.data_sd, self.config.data_td, self.config.embed, self.config.model_sd, self.config.model_td, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm
        self.mode = self.exp.mode.lower()  # 选择模式，默认条件下是test_llm_layer_split

        config.seed = int(config.seed or 2023)  # 随机数种子
        seeding(config.seed)  # 设置随机数种子

        self.init_pigmento()  # 定制打印输出格式

        pnt('START TIME:', datetime.datetime.now())  # 开始时间
        # print command line arguments
        pnt('python ', ' '.join(sys.argv))  # sys.argv是传给脚本的参数，包括.py文件以及各项要求输入的参数
        pnt(json.dumps(Obj.raw(self.config), indent=4))  # 输出JSON格式的config，为了美观

        # Meta是来自/loader/meta.py 的一个class
        Meta.device = self.get_device()  # 设置device
        Meta.simple_dev = self.exp.policy.simple_dev  # 默认为False
        # Setting.dataset = self.data.dataset

        # Controller是来自/loader/controller_sd.py 的一个class，包含了需要用到的几乎所有信息
        self.controller_sd = Controller(
            data=self.data_sd,
            embed=self.embed,
            model=self.model_sd,
            exp=self.exp,
        )  # 数据集包含在self.controller_sd.sets中

        # self.legommender_sd = self.controller_sd.legommender.to(Meta.device)  # 包括了News encoder, User encoder等最重要的部件
        # pretrained_parameters_sd0, other_parameters_sd0 = self.legommender_sd.get_parameters()

        self.controller_td = Controller(
            data=self.data_td,
            embed=self.embed,
            model=self.model_td,
            exp=self.exp,
            iftd=True,
        )  # 数据集包含在self.controller_td.sets中

        # self.legommender_td = self.controller_td.legommender.to(Meta.device)  # 包括了News encoder, User encoder等最重要的部件
        # pretrained_parameters_td0, other_parameters_td0 = self.legommender_td.get_parameters()

        print(f"device = {Meta.device}")

        self.controller_td.legommender.load_state_dict(self.controller_sd.legommender.state_dict())

        self.legommender_sd = self.controller_sd.legommender.to(Meta.device)  # 包括了News encoder, User encoder等最重要的部件
        self.resampler_sd = self.controller_sd.resampler  # item_cache，这是经由natural concator之后的完全由tokens所组成的序列；user inputer，包括了train set的behavior.tsv的数据
        self.cacher_sd = self.legommender_sd.cacher
        self.cacher_sd.activate(config.fast_eval)
        self.legommender_td = self.controller_td.legommender.to(Meta.device)  # 包括了News encoder, User encoder等最重要的部件
        self.resampler_td = self.controller_td.resampler  # item_cache，这是经由natural concator之后的完全由tokens所组成的序列；user inputer，包括了train set的behavior.tsv的数据
        self.cacher_td = self.legommender_td.cacher
        self.cacher_td.activate(config.fast_eval)

        # 同步更新模型参数
        self.legommender_sd.item_encoder.additive_attention = self.legommender_td.item_encoder.additive_attention
        self.legommender_sd.item_encoder.linear = self.legommender_td.item_encoder.linear
        self.legommender_sd.item_encoder.transformer = self.legommender_td.item_encoder.transformer
        self.legommender_sd.user_encoder.additive_attention = self.legommender_td.user_encoder.additive_attention

        self.load_path = self.parse_load_path()

        pnt(self.controller_sd.depots.a_depot()[0])
        pnt(self.controller_td.depots.a_depot()[0])
        pnt(Structure().analyse_and_stringify(self.controller_sd.sets.a_set()[0]))
        pnt(Structure().analyse_and_stringify(self.controller_td.sets.a_set()[0]))

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

    """pigmento是一个Qijiong Liu自己做的库，主要用于customize打印输出的各种格式"""

    def init_pigmento(self):
        pigmento.add_time_prefix()  # 增加时间前缀，用于显示运行的时间
        pigmento.add_log_plugin(self.exp.log)  # 每次print的时候，保存到日志文件中
        pigmento.add_dynamic_color_plugin()  # 对不同的class，自动切换不同的颜色来print
        pnt.set_display_mode(  # 定制输出格式，不显示method_name，显示class_name
            display_method_name=False,
            display_class_name=True,
        )

    def load(self, path):
        while True:
            pnt(f"load model from exp {path}")
            try:
                state_dict = torch.load(path, map_location=Meta.device)  # 记录model, optimizer, scheduler的各项信息
                break
            except Exception as e:
                if not self.exp.load.wait_load:
                    raise e
                time.sleep(60)

        # compatible to old version where each operator are wrapped with an encoder
        model_ckpt = dict()
        for key, value in state_dict['model'].items():  # 记录模型的各层参数，以及初始的嵌入层等等
            model_ckpt[key.replace('operator.', '')] = value

        self.legommender_td.load_state_dict(model_ckpt, strict=self.exp.load.strict)
        if not self.exp.load.model_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def parse_load_path(self):
        if not self.exp.load.save_dir:
            return

        save_dir = os.path.join(self.exp.dir, self.exp.load.save_dir)
        epochs = Obj.raw(self.exp.load.epochs)
        if not epochs:
            epochs = json.load(open(os.path.join(save_dir, 'candidates.json')))
        elif isinstance(epochs, str):
            epochs = eval(epochs)
        assert isinstance(epochs, list), ValueError(f'fail loading epochs: {epochs}')

        return [os.path.join(save_dir, f'epoch_{epoch}.bin') for epoch in epochs]

    """用于获取device的信息"""

    def get_device(self):
        cuda = self.config.cuda
        if cuda in ['-1', -1] or cuda is False:
            pnt('choose cpu')
            return 'cpu'
        if isinstance(cuda, int) or isinstance(cuda, str):
            pnt(f'User select cuda {cuda}')  # 选择cuda
            # return f"cuda:{cuda}"
            cuda = eval(f'[{cuda}]') if isinstance(cuda, str) else cuda
            device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
            # return torch.cuda.device(cuda)
            return device
        return GPU.auto_choose(torch_format=True)

    @staticmethod
    def log_interval(epoch, step, loss):
        pnt(f'[epoch {epoch}] step {step}, loss {loss:.4f}')

    @staticmethod
    def log_epoch(epoch, results):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        pnt(f'[epoch {epoch}] {line}')

    def train(self) -> int:
        monitor_kwargs = Obj.raw(self.exp.store)

        dev_func = self.dev
        if Meta.simple_dev:
            monitor_kwargs['maximize'] = False
            dev_func = self.simple_evaluate
            pnt('activate simple dev mode')

        monitor = Monitor(
            save_dir=self.exp.dir,
            **monitor_kwargs,
        )
        ########################################################
        train_steps = len(self.controller_td.sets.train_set) // self.exp.policy.batch_size
        ########################################################
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1  # 条件触发

        loader_sd = self.controller_sd.get_loader(
            Phases.train).train()  # DataLoader，其中cacher的item存储了BERT第layer+1层的一个输出结果
        loader_td = self.controller_td.get_loader(
            Phases.train).train()  # DataLoader，其中cacher的item存储了BERT第layer+1层的一个输出结果
        self.optimizer.zero_grad()  # 手动清零梯度，防止梯度累计，从而造成梯度爆炸

        # pretrained_parameters_sd1, other_parameters_sd1 = self.legommender_sd.get_parameters()
        # pretrained_parameters_td1, other_parameters_td1 = self.legommender_td.get_parameters()

        loss1_list = []
        loss2_list = []
        loss3_list = []
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):  # 从0到50
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.legommender_sd.train()  # 设置为训练模式，这里最终调用的实际上是torch.nn.Moudle中的train()函数
            self.legommender_td.train()  # 设置为训练模式，这里最终调用的实际上是torch.nn.Moudle中的train()函数
            loader_sd.train()  # 也是一堆训练模式的设置
            loader_td.train()  # 也是一堆训练模式的设置
            # 一个batch中包含了history(nid形式), candidates(key为nid，1+K个), click(batch_size个), imp(batch_size个), uid((batch_size个))，已经buffle过，都是ID形式
            # for step, batch_sd in enumerate(tqdm(loader_sd, disable=self.disable_tqdm)): #不同epoch下的每个step的batch是不同的，但是同个epoch在每次运行下的batch都是一样的，这是为了reproducibility
            for step, (batch_sd, batch_td) in enumerate(tqdm(zip(loader_sd, loader_td), disable=self.disable_tqdm,
                                                             total=min(len(loader_sd), len(loader_td)))):
                ############################################
                # with open("/home/zhangyy/code/Lego-title_A40/batch_small_sdtd.pkl", "rb") as f:
                #     batch_sd = pickle.load(f)
                # with open("/home/zhangyy/code/Lego-title_A40/batch_td2.pkl", "rb") as f:
                #     batch_td = pickle.load(f)
                loss1 = self.legommender_sd(batch=batch_sd)  # ENG上的loss
                loss2 = self.legommender_td(batch=batch_td)  # Lingual上的loss
                loss3 = self.legommender_td(batch=batch_td, contrastive=True)  # 对比学习的loss
                loss4 = self.legommender_td(batch=batch_td, contrastive=True, rank=True)  # 对比学习的loss

                alpha = 1
                beta = 1
                gamma = 1
                delta = 1
                loss = (alpha * loss1 + beta * loss2 + gamma * loss3 + delta * loss4) / (alpha + beta + gamma + delta)
                loss.backward()  # 计算所有需要训练的param的grad，并记录在self.optimizer.param_groups中
                self.legommender_td.item_encoder.ifeng = False

                # self.optimizer.step()  # 更新参数
                # self.scheduler.step()  # 更新优化器，实际上就是改变一下学习率
                # self.optimizer.zero_grad()  # 清除梯度
                #
                # pretrained_parameters_sd2, other_parameters_sd2 = self.legommender_sd.get_parameters()
                # pretrained_parameters_td2, other_parameters_td2 = self.legommender_td.get_parameters()
                #
                # self.log_interval(epoch, 0, loss.item())
                # dev_results, monitor_metric = dev_func()  # dev_results是个OrderedDict，不过实际上key只有GAUC，而monitor_metric中是具体的值
                # self.log_epoch(epoch, dev_results)  # 打印结果

                accumulate_step += 1  # gradient accumulation，相当于使用了
                if accumulate_step == accumulate_batch:
                    self.optimizer.step()  # 更新参数
                    self.scheduler.step()  # 更新优化器，实际上就是改变一下学习率
                    self.optimizer.zero_grad()  # 清除梯度
                    accumulate_step = 0
                # 记录日志，即epoch, step, loss.item
                if self.exp.policy.check_interval:
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, loss.item())  # [epoch 0] step 0, loss 1.6152
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, loss.item())  # [epoch 0] step 0, loss 1.6152
                # 控制一下step的上限，但是为什么要这样子呢？
                if self.exp.policy.epoch_batch:
                    if self.exp.policy.epoch_batch < 0:  # step part
                        if step > max(train_steps // (-self.exp.policy.epoch_batch), 1):
                            break
                    else:
                        if step > self.exp.policy.epoch_batch:
                            break
                ############################################

            # 是在dev数据集上算的GAUC，而且会根据batch_size来分

            # pretrained_parameters_sd2, other_parameters_sd2 = self.legommender_sd.get_parameters()
            # pretrained_parameters_td2, other_parameters_td2 = self.legommender_td.get_parameters()

            dev_results, monitor_metric = dev_func()  # 三条bar都在这，dev_results是个OrderedDict，不过实际上key只有GAUC，而monitor_metric中是具体的值
            self.log_epoch(epoch, dev_results)  # [epoch 0] GAUC 0.5210

            state_dict = dict(  # 关于model，optimizer，scheduler的各项参数
                model=self.legommender_td.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
            )
            early_stop = monitor.push(
                epoch=epoch,
                metric=monitor_metric,
                state_dict=state_dict,
            )
            if early_stop == -1:
                return monitor.get_best_epoch()

        # with open('/private/task/zhangyy/code/Lego-CL/other_saving/loss1.pkl', 'wb') as f:
        # pickle.dump(loss1_list,f)
        # with open('/private/task/zhangyy/code/Lego-CL/other_saving/loss2.pkl', 'wb') as f:
        # pickle.dump(loss2_list,f)
        # with open('/private/task/zhangyy/code/Lego-CL/other_saving/loss3.pkl', 'wb') as f:
        # pickle.dump(loss3_list,f)

        pnt('Training Ended')
        monitor.export()

        return monitor.get_best_epoch()

    def dev(self):
        assert self.exp.store.metric
        loader = self.controller_td.get_loader(Phases.dev).eval()

        results = self.evaluate(loader, metrics=[self.exp.store.metric])
        return results, results[self.exp.store.metric]

    def test_fake(self):
        self.legommender_td.eval()
        loader = self.controller_td.get_loader(Phases.test).test()
        loader.dataset.timer.clear()

        score_series, label_series, group_series, fake_series = [], [], [], []
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                scores = self.legommender_td(batch=batch)
            labels = batch[self.controller_td.column_map.label_col].tolist()
            groups = batch[self.controller_td.column_map.group_col].tolist()
            fakes = batch[self.controller_td.column_map.fake_col].tolist()
            score_series.extend(scores.cpu().detach().tolist())
            label_series.extend(labels)
            group_series.extend(groups)
            fake_series.extend(fakes)

        df = pd.DataFrame(dict(
            score=score_series,
            label=label_series,
            group=group_series,
            fake=fake_series,
        ))
        groups = df.groupby('fake')
        pnt('group by fake done')

        for fake, group in groups:
            pnt('inactive users' if fake else 'active users')
            pool = MetricPool.parse(self.exp.metrics)
            results = pool.calculate(group['score'].tolist(), group['label'].tolist(), group['group'].tolist())
            for metric in results:
                pnt(f'{metric}: {results[metric]:.4f}')

    def mind_large_evaluate(self, loader):
        self.legommender_td.eval()

        # group_series = submission.depot.data[self.config_manager.column_map.group_col].tolist()
        # item_series = submission.depot.data[self.config_manager.column_map.candidate_col].tolist()
        # score_series = [random.random() for _ in range(len(submission.depot))]
        item_col, group_col = self.controller_td.column_map.candidate_col, self.controller_td.column_map.group_col
        score_series, col_series = self.base_evaluate(loader, cols=[item_col, group_col])
        item_series, group_series = col_series[item_col], col_series[group_col]
        # item_series = [v[0] for v in item_series]

        loader.dataset.timer.summarize()

        submission = Submission(
            depot=self.controller_td.depots[Phases.test],
            column_map=self.controller_td.column_map,
        )

        export_dir = submission.run(
            scores=score_series,
            groups=group_series,
            items=item_series,
            model_name=self.model_td.name,
        )

        pnt(f'export to {export_dir}')

    def test(self):
        loader = self.controller_td.get_loader(Phases.test).test()  # 加载数据集，包括已经cache过的user和item的embedding

        if self.config.mind_large_submission:
            return self.mind_large_evaluate(loader)

        results = self.evaluate(loader, metrics=self.exp.metrics)  # exp.metrics中记录了需要计算的指标
        for metric in results:  # 遍历打印计算的各个metric结果
            pnt(f'{metric}: {results[metric]:.4f}')

    def train_get_user_embedding(self):
        self.controller_td.get_loader(Phases.train).test()
        assert self.cacher.user.cached, 'fast eval not enabled'
        user_embeddings = self.cacher.user.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'user_embeddings.npy')
        pnt(f'store user embeddings to {store_path}')
        np.save(store_path, user_embeddings)

    def train_get_item_embedding(self):
        self.cacher.item.cache(self.resampler.item_cache)
        item_embeddings = self.cacher.item.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'item_embeddings.npy')
        pnt(f'store item embeddings to {store_path}')
        np.save(store_path, item_embeddings)

    def base_evaluate(self, loader, cols):
        score_series = torch.zeros(len(loader.dataset),
                                   dtype=torch.float32)  # len(loader.dataset)是测试数据中所有形如N1234-0的impression的个数
        col_series = {col: torch.zeros(len(loader.dataset), dtype=torch.long) for col in
                      cols}  # 包含两个字典，key分别是click和imp，然后初始化为一个len(loader.dataset)的全0的tensor

        index = 0
        for step, batch in enumerate(tqdm(loader,
                                          disable=self.disable_tqdm)):  # batch包含nid, click, imp, uid，其中click标识是否点击，imp是每条behavior数据的index，一条数据对应同一个imp
            with torch.no_grad():
                scores = self.legommender_td(batch=batch)  # 计算一个batch中，每个user-news配对的分数
                if scores.dim() == 2:  # 转化成一维的
                    scores = scores.squeeze(1)

            batch_size = scores.size(0)  # batch的大小
            for col in cols:  # 给一个batch的col_series赋值
                if batch[col].dim() == 2:
                    col_series[col][index:index + batch_size] = batch[col][:, 0]
                else:
                    col_series[col][index:index + batch_size] = batch[col]
            score_series[index:index + batch_size] = scores.cpu().detach()  # 给一个batch的score_series
            index += batch_size  # 更新index

        return score_series, col_series

    def evaluate(self, loader, metrics):
        pool = MetricPool.parse(metrics)  # 得到需要计算的metrics list
        self.legommender_td.eval()

        label_col, group_col = self.controller_td.column_map.label_col, self.controller_td.column_map.group_col  # label_col是表示是否点击的01，group_col是impression
        score_series, col_series = self.base_evaluate(loader, cols=[label_col, group_col])
        label_series, group_series = col_series[label_col], col_series[
            group_col]  # label_series记录每个user-news对得到点击情况（0 or 1），group_series记录每个user-news所属的分组，即同一条behavior数据是同一个分组

        results = pool.calculate(score_series, label_series, group_series)  # 计算各项指标
        return results

    def simple_evaluate(self):
        loader = self.controller_td.get_loader(Phases.dev).eval()
        total_loss = Meaner()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.legommender_td(batch=batch)
            total_loss.add(loss.item())

        total_loss = total_loss.mean()
        return dict(loss=total_loss), total_loss

    """
    运行训练过程，此时的self.legommender已经包含了item encoder和user encoder，其中item encoder包含了hidden_weights，对应config.item_config中的layer_split
    对于BERT而言，pretrained_parameters至少包含6个，分别是position_embeddings，token_type_embeddings，LayerNorm所需要的weight(缩放)和bais(位移)，以及在所有层结束后的pooler的weight和bias
    pooler一般针对[CLS]处理，提炼和压缩[CLS]的表示来作为最终的可以用于下游任务表示的整句话的embedding
    如果需要LoRA微调的话则还会分别包含针对query和value的两个LoRA矩阵，原权重矩阵+BA后得到经过LoRA微调后的矩阵，相当于BERT中的一个Transformer层有4个关于LoRA的矩阵
    other_parameters主要包括一些固定token的embedding，深度学习推荐模型的item encoder和user encoder的网络参数等
    """

    def train_runner(self):
        if self.legommender_td.config.use_item_content and self.exp.policy.item_lr:
            pnt('split item pretrained encoder parameters')
            pnt('pretrained lr:', self.exp.policy.item_lr)
            pnt('other lr:', self.exp.policy.lr)
            pretrained_parameters, other_parameters = self.legommender_td.get_parameters()  # 获取需要训练的参数
            self.optimizer = torch.optim.Adam([
                {'params': pretrained_parameters, 'lr': self.exp.policy.item_lr},
                {'params': other_parameters, 'lr': self.exp.policy.lr}  #
            ])
        else:
            pnt('use single lr:', self.exp.policy.lr)
            self.optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.legommender_td.parameters()),
                lr=self.exp.policy.lr
            )

            for name, p in self.legommender_td.named_parameters():  # type: str, torch.Tensor
                if p.requires_grad:
                    pnt(name, p.data.shape)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            ########################################################
            num_training_steps=len(
                self.controller_td.sets.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
            ########################################################
        )

        if self.load_path:  # 如果有checkpoint的模型则加载这个checkpoint的模型
            self.load(self.load_path[0])
        return self.train()

    def test_runner(self):
        self.test()

    def iter_runner(self, handler):
        if self.load_path:
            for path in self.load_path:
                self.load(path)
                handler()
        else:
            handler()

    def test_size(self):
        named_parameters = list(self.legommender_td.named_parameters())
        # filter out the parameters that don't require a gradient
        named_parameters = [(name, p) for (name, p) in named_parameters if p.requires_grad]
        # list of (name, parameter) pairs
        for (name, p) in named_parameters:
            pnt(name, p.data.shape)
        num_params = sum([p.numel() for (_, p) in named_parameters])
        # to a million
        num_params /= 1e6
        pnt(f'Number of parameters: {num_params:.2f}M')

    """对应ONCE中readme的Training Preparation, 为的是提前获取在layers层的hidden_states，这样后续在fine-tuning的时候就可以节约计算资源，俗称caching"""
    """显然这部分是不涉及训练的，最终结果就是以npy格式保存的所需要的layers层的输出hidden_states结果，在MINDtiny_bert中就是645*30*768，一个layer对应一个npy文件"""

    def test_llm_layer_split(self):
        item_encoder = self.legommender_td.item_encoder  # type: BaseLLMOperator
        assert isinstance(item_encoder, BaseLLMOperator), 'llama operator not found'

        pager = LLMSplitPager(
            inputer=item_encoder.inputer,
            layers=Obj.raw(self.exp.store.layers),  # llama_split.yaml里面定义的那些layers，用list表示
            hidden_size=item_encoder.config.embed_hidden_size,  # LLM中的维度
            contents=self.resampler_td.item_cache,  # resample.item_cache中存储了item的natural concator的结果，用token序列表示
            model=item_encoder.get_all_hidden_states,  #
            page_size=self.config.page_size,
        )  # pager中的final_features的维度是 （leyers数量）*（news数量）*（content长度）*（embed_hidden_size）

        pager.run()  # 得到final_features和final_masks
        pager.store(self.exp.store.dir)  # 存储所需要的layers的hidden_states，一个npy文件是（news数量）*（content长度）*（embed_hidden_size）

    def run(self):
        # params = Obj.raw(self.exp.params)
        # if not isinstance(params, dict):
        #     params = dict()
        #
        # mode_worker = self.mode_hub(self.mode)
        # if mode_worker.load_model:

        if self.mode == 'train':  # 训练模式
            self.train_runner()
        elif self.exp.mode == 'test':  # 测试模式
            self.iter_runner(self.test_runner)
            # self.test()
        elif self.exp.mode == 'test_fake':
            self.iter_runner(self.test_fake)
        elif self.mode == 'train_test':  # tt，Training and Tesing
            epoch = self.train_runner()  # 训练结束后最佳的一个epoch，注意是从0开始计数的
            self.load(os.path.join(self.exp.dir, f'epoch_{epoch}.bin'))  # 加载训练结束后的model，用于后续测试
            self.test_runner()
        elif self.mode == 'test_size':
            self.test_size()
        elif self.mode == 'test_llm_layer_split':  # Training Preparation
            self.test_llm_layer_split()
        elif self.mode == 'train_get_user_embedding':
            if self.load_path:
                self.load(self.load_path[0])
            self.train_get_user_embedding()
        elif self.mode == 'train_get_item_embedding':
            self.load(self.load_path[0])
            self.train_get_item_embedding()

        # self.mode_hub(self.mode)()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data_sd', 'data_td', 'model_sd', 'model_td', 'exp'],
        default_args=dict(
            embed='config/embed/null.yaml',
            warmup=0,
            fast_eval=True,
            simple_dev=False,
            batch_size=64,
            acc_batch=1,
            lora=1,
            lora_r=32,
            lr=0.0001,
            item_lr=0.00001,
            mind_large_submission=False,
            hidden_size=64,
            epoch_batch=0,
            max_item_batch_size=0,
            page_size=256,
            patience=10,
            epoch_start=0,
            frozen=True,
            load_path=None,
            seed=2024,
            cuda=1
        ),
        makedirs=[
            'exp.dir',
        ]
    ).parse()  # 读取各种参数

    worker = Worker(config=configuration)  # 创建一个名为Worker的class
    worker.run()
