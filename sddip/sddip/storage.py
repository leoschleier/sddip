from abc import ABC, abstractmethod
import pandas as pd


class ResultStorage:

    def __init__(self, result_keys:list =[]):
        self.result_keys = result_keys
        self.index_names = ("i", "k", "t")
        self.results = {}

    def add_result(self, iteration_index: int, sample_index: int, stage_index: int, results:dict):
        self.results[iteration_index, sample_index, stage_index] = results

    def get_result(self, iteration_index: int, sample_index: int, stage_index: int):
        return self.results[iteration_index, sample_index, stage_index]

    def get_stage_result(self, stage_index: int):
        df = self.to_dataframe()
        return df.query("t == %i"%(stage_index)).to_dict("list")

    def to_dataframe(self):
        df = pd.DataFrame.from_dict(self.results, orient="index")
        return df.rename_axis(self.index_names)

    def create_empty_result_dict(self):
        return {key: [] for key in self.result_keys}
