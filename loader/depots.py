import json
from typing import Dict

from pigmento import pnt

from loader.column_map import ColumnMap
from loader.depot.caching_depot import CachingDep
from loader.depot.depot_hub import DepotHub
from loader.meta import Phases, Meta


class Depots:
    def __init__(self, user_data, modes: set, column_map: ColumnMap):
        self.column_map = column_map #列映射关系

        self.train_depot = self.dev_depot = self.test_depot = None
        # 虽然看不懂发生了什么，但是结果是self.xxx_depot中将包含来自path的一堆数据
        if Phases.train in modes: #模式为train
            self.train_depot = DepotHub.get(user_data.depots.train.path, filter_cache=user_data.filter_cache)
        if Phases.dev in modes: #模式为dev
            self.dev_depot = DepotHub.get(user_data.depots.dev.path, filter_cache=user_data.filter_cache)
        if Phases.test in modes: #模式为test
            self.test_depot = DepotHub.get(user_data.depots.test.path, filter_cache=user_data.filter_cache)

        self.fast_eval_depot = self.create_fast_eval_depot(user_data.depots.dev.path, column_map=column_map)

        self.depots = {
            Phases.train: self.train_depot, #训练数据集
            Phases.dev: self.dev_depot, #验证数据集
            Phases.test: self.test_depot, #测试数据集
            Phases.fast_eval: self.fast_eval_depot, #快速验证数据集
        }  # type: Dict[str, CachingDep]

        if user_data.union: #如果union有值，正常情况是一个长度为2的列表，一个是user的路径，一个是neg的路径
            for depot in self.depots.values(): #遍历上面self.depots的四个values
                if not depot:
                    continue
                depot.union(*[DepotHub.get(d) for d in user_data.union])

        if user_data.allowed:
            allowed_list = json.load(open(user_data.allowed))
            for phase in self.depots:
                depot = self.depots[phase]
                if not depot:
                    continue
                sample_num = len(depot)
                super(CachingDep, depot).filter(lambda x: x in allowed_list, col=depot.id_col)
                pnt(f'Filter {phase} phase with allowed list, sample num: {sample_num} -> {len(depot)}')

        if user_data.filters: #过滤掉重复项
            for col in user_data.filters: # 只有一个history
                for filter_str in user_data.filters[col]:
                    filter_func_str = f'lambda x: {filter_str}'
                    for phase in [Phases.train, Phases.dev, Phases.test]:
                        depot = self.depots[phase]
                        if not depot:
                            continue
                        sample_num = len(depot)
                        depot.filter(filter_func_str, col=col)
                        pnt(f'Filter {col} with {filter_str} in {phase} phase, sample num: {sample_num} -> {len(depot)}')

        for phase in [Phases.train, Phases.dev, Phases.test]:
            filters = user_data.depots[phase].filters
            depot = self.depots[phase]
            if not depot:
                continue
            for col in filters:
                for filter_str in filters[col]:
                    filter_func_str = f'lambda x: {filter_str}'
                    depot.filter(filter_func_str, col=col)
                    pnt(f'Filter {col} with {filter_str} in {phase} phase, sample num: {len(depot)}')

    @staticmethod
    def create_fast_eval_depot(path, column_map: ColumnMap):
        user_depot = CachingDep(path)
        user_num = user_depot.cols[column_map.user_col].voc.size
        user_depot.reset({
            user_depot.id_col: list(range(user_num)),
            column_map.candidate_col: [[0] for _ in range(user_num)],
            column_map.label_col: [[0] for _ in range(user_num)],
            column_map.user_col: list(range(user_num)),
            column_map.group_col: list(range(user_num)),
        })
        return user_depot

    def negative_filter(self, col):
        phases = [Phases.train]
        if Meta.simple_dev:
            phases.append(Phases.dev)

        for phase in phases:
            depot = self.depots[phase]
            if not depot:
                continue

            sample_num = len(depot)
            depot.filter('lambda x: x == 1', col=col)
            pnt(f'Filter {col} with x==1 in {phase} phase, sample num: {sample_num} -> {len(depot)}')

    def __getitem__(self, item):
        return self.depots[item]

    def a_depot(self):
        return self.train_depot or self.dev_depot or self.test_depot
