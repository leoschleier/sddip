import os

import pandas as pd


class ResultStorage:
    def __init__(
        self, result_keys: list | None = None, label="results"
    ) -> None:
        if result_keys is None:
            result_keys = []
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
    ) -> None:
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
        """Create a pandas DataFrame from the results dictionary."""
        if self.results:
            results = self.results
        else:
            results = {("n/a", "n/a", "n/a"): {"n/a": "n/a"}}

        df = pd.DataFrame.from_dict(results, orient="index")
        return df.rename_axis(self.index_names)

    def create_empty_result_dict(self):
        return {key: [] for key in self.result_keys}

    def export_results(self, results_dir: str) -> None:
        df = self.to_dataframe()
        df.to_csv(os.path.join(results_dir, f"{self.label}.csv"), sep="\t")
