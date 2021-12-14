import pandas as pd


class SolutionStorage:

    def __init__(self):
        self.index_names = ("i", "k", "t")
        self.solutions = {}

    def add_solution(self, iteration_index: int, sample_index: int, stage_index: int, solution:dict):
        self.solutions[iteration_index, sample_index, stage_index] = solution

    def get_solution(self, iteration_index: int, sample_index: int, stage_index: int):
        return self.solutions[iteration_index, sample_index, stage_index]

    def to_dataframe(self):
        df = pd.DataFrame.from_dict(self.solutions, orient="index")
        return df.rename_axis(self.index_names)