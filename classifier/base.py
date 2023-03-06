import abc

import metrics
import model


class BaseClassifier(abc.ABC):
    def fit(self, train_set: model.DataSet, dev_set: model.DataSet) -> metrics.F1Metrics:
        raise NotImplementedError()

    def score(self, test_set: model.DataSet) -> metrics.F1Metrics:
        raise NotImplementedError()
