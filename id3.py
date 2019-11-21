from math import log2
import pandas
from ID3.decision_tree import DecisionTree


class ID3:

    def __init__(self, in_attr_list, out_attr):
        self.in_attr_list = in_attr_list
        self.out_attr = out_attr

        self.tree = DecisionTree()

    def generate_decision_tree(self, data):
        self.id3(self.in_attr_list, data)
        return self.tree.get_tree()

    def id3(self, in_attr_list: list, data: pandas.DataFrame, grouping_value="", ancestors=[]):
        if data.empty:
            self.tree.add_to_tree(ancestors, 'Failed')
            return

        if data[self.out_attr].unique().size == 1:
            node = data[self.out_attr].unique()[0]
            self.tree.add_to_tree(ancestors, node)
            return

        if not in_attr_list:
            node = data[self.out_attr].value_counts().idxmax()
            self.tree.add_to_tree(ancestors, node)
            return

        largest_ig_attr = self.__largest_ig_attr(in_attr_list, self.out_attr, data)
        self.tree.add_to_tree(ancestors, largest_ig_attr, True)

        temp_in_attr_list = in_attr_list.copy()
        temp_in_attr_list.remove(largest_ig_attr)
        for grouping_value, group in data.groupby([largest_ig_attr]):
            self.id3(
                in_attr_list   = temp_in_attr_list,
                data           = group.drop([largest_ig_attr], axis=1),
                grouping_value = grouping_value,
                ancestors      = ancestors + [(largest_ig_attr, grouping_value)]
            )

        return

    @classmethod
    def __largest_ig_attr(cls, in_attr_list, out_attr, data):
        highest_ig_attr = in_attr_list[0]
        highest_ig = cls.information_gain(highest_ig_attr, out_attr, data)

        for attr in in_attr_list[1:]:
            cur_ig = cls.information_gain(attr, out_attr, data)
            if highest_ig < cur_ig:
                highest_ig_attr = attr
                highest_ig = cur_ig

        return highest_ig_attr

    @classmethod
    def information_gain(cls, in_attr, out_attr, data):
        return cls.entropy(data[out_attr]) - cls.in_attr_entropy(in_attr, out_attr, data)

    @staticmethod
    def entropy(data):
        row_count = data.count()
        entropy = 0

        for value, value_count in data.value_counts().items():
            probability = value_count / row_count
            entropy -= probability * log2(probability)

        return entropy

    @staticmethod
    def in_attr_entropy(in_attr, out_attr, data):
        row_count = len(data.index)
        entropy = 0

        for group_name, group in data.groupby([in_attr]):
            group_size = len(group.index)
            entropy_acc = 0

            for subgroup_name, subgroup in group.groupby([out_attr]):
                probability = len(subgroup.index) / group_size
                entropy_acc -= probability * log2(probability)

            entropy_acc *= group_size / row_count
            entropy += entropy_acc

        return entropy

    def classify(self, data):
        if not self.tree.get_tree():
            raise Exception('Decision tree not generated.')

        result = []
        for index, row in data.iterrows():
            result.append(self.classify_row(row, self.tree.get_tree()))

        return result

    def classify_row(self, row, tree):
        attr = next(iter(tree))
        row_val = row[attr]
        decision = tree[attr][row_val]

        if isinstance(decision, dict):
            return self.classify_row(row, tree[attr][row_val])
        else:
            return decision