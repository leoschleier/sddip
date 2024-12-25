import os
from datetime import datetime

import pandas as pd

from .. import config


class ResultsManager:
    def __init__(self):
        pass

    def create_results_dir(self, dir_label: str) -> str:
        start_time_str = datetime.today().strftime("%Y_%m_%d__%H_%M_%S")
        dir_name = f"{dir_label}_{start_time_str}"
        results_dir = config.RESULTS_DIR / dir_name
        os.makedirs(results_dir)

        return results_dir


class ResultStorage:
    def __init__(self, result_keys: list = [], label="results"):
        self.result_keys = result_keys
        self.label = label
        self.index_names = ("i", "k", "t")
        self.results = {}

    def add_result(
        self,
        iteration_index: int,
        sample_index: int,
        stage_index: int,
        results: dict,
    ):
        self.results[iteration_index, sample_index, stage_index] = results

    def get_result(
        self, iteration_index: int, sample_index: int, stage_index: int
    ):
        return self.results[iteration_index, sample_index, stage_index]

    def get_stage_result(self, stage_index: int):
        df = self.to_dataframe()
        return df.query("t == %i" % (stage_index)).to_dict("list")

    def get_iteration_result(self, iteration: int):
        df = self.to_dataframe()
        return df.query("i == %i" % (iteration)).to_dict("list")

    def to_dataframe(self):
        df = pd.DataFrame.from_dict(self.results, orient="index")
        return df.rename_axis(self.index_names)

    def create_empty_result_dict(self):
        return {key: [] for key in self.result_keys}

    def export_results(self, results_dir: str):
        df = self.to_dataframe()
        df.to_csv(os.path.join(results_dir, f"{self.label}.csv"), sep="\t")
