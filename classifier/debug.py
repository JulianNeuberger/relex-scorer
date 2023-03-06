import typing

import metrics
import model
from classifier import base


class DebugClassifier(base.BaseClassifier):
    def __init__(self):
        self.training_variance = 0.0
        self.training_imbalance = 0.0

    @staticmethod
    def _get_var_and_imbalance_from_set(dataset: model.DataSet):
        relation_count = {}
        relation_variance_stats: typing.Dict[str, typing.Tuple[typing.Set, int]] = {}
        for sample in dataset.samples:
            for relation in sample.relations:
                if relation.type not in relation_count:
                    relation_count[relation.type] = 0
                    relation_variance_stats[relation.type] = (set(), 0)
                relation_count[relation.type] += 1
                tokens = [
                    sample.tokens[i]
                    for i in relation.head.document_level_token_indices + relation.tail.document_level_token_indices
                ]
                for token in tokens:
                    relation_variance_stats[relation.type][0].add(token)

                relation_variance_stats[relation.type] = (relation_variance_stats[relation.type][0], relation_variance_stats[relation.type][1] + len(tokens))
        imbalance = 1.0 - min(relation_count.values()) / max(relation_count.values())
        variance = 0.0
        for unique_tokens, total_num_tokens in relation_variance_stats.values():
            variance += len(unique_tokens) / total_num_tokens
        variance /= len(relation_variance_stats)

        return imbalance, variance

    @staticmethod
    def metrics_from_stats(train_imbalance, test_imbalance, train_variance, test_variance) -> metrics.F1Metrics:
        p = (train_variance - test_variance + 1) / 2.0
        r = train_imbalance
        f1 = 2 * p * r / (p + r)
        return metrics.F1Metrics(precision=p, recall=r, f1=f1)

    def fit(self, train_set: model.DataSet, dev_set: model.DataSet) -> metrics.F1Metrics:
        self.training_imbalance, self.training_variance = self._get_var_and_imbalance_from_set(train_set)
        dev_imbalance, dev_variance = self._get_var_and_imbalance_from_set(dev_set)

        return self.metrics_from_stats(self.training_imbalance, dev_imbalance, self.training_variance, dev_variance)

    def score(self, test_set: model.DataSet) -> metrics.F1Metrics:
        test_imbalance, test_variance = self._get_var_and_imbalance_from_set(test_set)
        return self.metrics_from_stats(self.training_imbalance, test_imbalance, self.training_variance, test_variance)
