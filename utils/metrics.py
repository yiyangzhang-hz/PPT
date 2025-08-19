from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Union, List

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score, ndcg_score, label_ranking_average_precision_score, f1_score


class Metric:
    name: str
    group: bool

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        pass

    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)

    def __str__(self):
        return self.name


class LogLoss(Metric):
    name = 'LogLoss'
    group = False

    def calculate(self, scores: list, labels: list):
        return log_loss(labels, scores)


class AUC(Metric):
    name = 'AUC'
    group = False

    def calculate(self, scores: list, labels: list):
        return roc_auc_score(labels, scores)


class GAUC(AUC):
    name = 'GAUC'
    group = True


class MRR(Metric):
    name = 'MRR'
    group = True

    def calculate(self, scores: list, labels: list):
        return label_ranking_average_precision_score([labels], [scores])


class F1(Metric):
    name = 'F1'
    group = False

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def calculate(self, scores: list, labels: list):
        scores = [int(score >= self.threshold) for score in scores]
        return f1_score(labels, scores)

    def __str__(self):
        return f'{self.name}@{self.threshold}'


class HitRatio(Metric):
    name = 'HitRatio'
    group = True

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        return int(1 in labels[:self.n])
        # candidates_set = set(scores[:self.n])
        # interaction = candidates_set.intersection(set(labels))
        # return int(bool(interaction))

    def __str__(self):
        return f'{self.name}@{self.n}'


class Recall(Metric):
    name = 'Recall'
    group = True

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        return sum(labels[:self.n]) * 1.0 / sum(labels)
        # candidates_set = set(scores[:self.n])
        # interaction = candidates_set.intersection(set(labels))
        # return len(interaction) * 1.0 / self.n

    def __str__(self):
        return f'{self.name}@{self.n}'


class NDCG(Metric):
    name = 'NDCG'
    group = True

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        return ndcg_score([labels], [scores], k=self.n)

    def __str__(self):
        return f'{self.name}@{self.n}'


class MetricPool:
    metric_list = [LogLoss, AUC, GAUC, F1, Recall, NDCG, HitRatio, MRR]
    metric_dict = {m.name.upper(): m for m in metric_list}

    def __init__(self, metrics):
        self.metrics = metrics  # type: List[Metric]
        self.values = OrderedDict()  # type: Dict[str, Union[list, float]]
        self.group = False

        for metric in self.metrics:
            self.values[str(metric)] = []
            self.group = self.group or metric.group

    @classmethod
    def parse(cls, metrics_config):
        metrics = []
        for m in metrics_config:
            at = m.find('@')
            argument = []
            if at > -1:
                m, argument = m[:at], [int(m[at+1:])]
            if m.upper() not in MetricPool.metric_dict:
                raise ValueError(f'Metric {m} not found')
            metrics.append(MetricPool.metric_dict[m.upper()](*argument))
        return cls(metrics)

    def calculate(self, scores, labels, groups, group_worker=5):
        if not self.metrics: #如果metrics为空，那么直接返回
            return {}
        # 建立一个表格，列表头groups用于记录所属的分组（每条behavior的全部impression为一个分组），scores用于记录每个user-news配对的分数，labels用于记录是否点击（0 or 1）
        df = pd.DataFrame(dict(groups=groups, scores=scores, labels=labels))

        groups = None
        if self.group:
            groups = df.groupby('groups') #按groups来分组

        for metric in self.metrics:
            if not metric.group:
                self.values[str(metric)] = metric(
                    scores=scores,
                    labels=labels,
                )
                continue

            tasks = []
            pool = Pool(processes=group_worker) #构建进程池，进程数量为group_worker
            for g in groups: #一个g[1]包含了一条behavior中的所有user-news配对的scores和对应的是否点击的labels
                group = g[1]
                g_labels = group.labels.tolist() #list形式的labels，即0/1表示的是否点击
                g_scores = group.scores.tolist() #list形式的scores，预测的分数，用来排序
                tasks.append(pool.apply_async(metric, args=(g_scores, g_labels))) #异步调用"metric"所对应的函数，参数就是args，即并行的计算指标
            pool.close() #关闭进程池，不再接受新的任务
            pool.join() #等待所有进程完成
            values = [t.get() for t in tasks] #所计算的每一条behaviors的某个指标
            self.values[str(metric)] = torch.tensor(values, dtype=torch.float).mean().item() #将所有behavior的平均结果作为得到的metric值
        return self.values

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)
