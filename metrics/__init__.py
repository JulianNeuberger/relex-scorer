import collections
import dataclasses
import typing
from typing import Iterable

import numpy as np

from metrics.parser import LabeledSample, BaseParser, TwoAreBetterThanOneParser, JointERParser, CasRelParser, \
    RSANParser, SpertParser, PFNCustomParser, MareParser
from metrics.spans import BaseMatcher, ExactRelationMatcher, BoundaryRelationMatcher, EntityMatcher

available_result_parsers: typing.Dict[str, BaseParser] = {
    'two-are-better-than-one': TwoAreBetterThanOneParser(),
    'joint-er': JointERParser(),
    'casrel': CasRelParser(),
    'rsan': RSANParser(),
    'spert': SpertParser(),
    'pfn': PFNCustomParser(),
    'mare': MareParser()
}


@dataclasses.dataclass
class F1Metrics:
    f1: float
    precision: float
    recall: float


def f1_micro_local(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    matches = [match_strategy.match(p) for p in predictions]
    n_true = sum([m.n_true for m in matches])
    n_pred = sum([m.n_pred for m in matches])
    n_ok = sum([m.n_ok for m in matches])

    precision = n_ok / n_pred if n_pred != 0.0 else 0.0
    recall = n_ok / n_true if n_true != 0.0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0.0 else 0.0

    return F1Metrics(
        f1=f1,
        precision=precision,
        recall=recall
    )


def f1_micro_global(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    matches = [match_strategy.match(p) for p in predictions]

    f1_scores = []
    p_scores = []
    r_scores = []
    for m in matches:
        precision = 0.0
        if m.n_pred == 0:
            if m.n_true == 0:
                precision = 1.0
        else:
            precision = m.n_ok / m.n_pred

        recall = 0.0
        if m.n_true == 0:
            if m.n_pred == 0:
                recall = 1.0
        else:
            recall = m.n_ok / m.n_true

        f1 = 0.0
        if precision + recall != 0.0:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)
        p_scores.append(precision)
        r_scores.append(recall)
    f1 = sum(f1_scores) / len(f1_scores)
    precision = sum(p_scores) / len(p_scores)
    recall = sum(r_scores) / len(r_scores)

    return F1Metrics(
        f1=f1,
        precision=precision,
        recall=recall
    )


def f1_macro_global(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    scores_by_relation_type: typing.Dict[str, typing.Tuple[float, float, float]] = collections.defaultdict(lambda: (0, 0, 0))

    for p in predictions:
        for rel in p.labels:
            # count the number of hits for this specific relation
            tmp = LabeledSample(labels=[rel], prediction=p.prediction, sample_id='')
            matches = match_strategy.match(tmp)

            n_gold, n_pred, n_ok = scores_by_relation_type[rel.tag]
            scores_by_relation_type[rel.tag] = (n_gold + matches.n_true, n_pred + matches.n_pred, n_ok + matches.n_ok)

    p_scores = []
    r_scores = []
    for rel_type, (n_gold, n_pred, n_ok) in scores_by_relation_type.items():
        p = 0.0
        if n_pred == 0:
            if n_gold == 0:
                p = 1.0
        else:
            p = n_ok / n_pred
        p_scores.append(p)

        r = 0.0
        if n_gold == 0:
            if n_pred == 0:
                r = 1.0
        else:
            r = n_ok / n_gold
        r_scores.append(r)

    macro_p = sum(p_scores) / len(p_scores)
    macro_r = sum(r_scores) / len(r_scores)
    macro_f1 = 0.0
    if macro_p + macro_r != 0.0:
        macro_f1 = 2 * macro_p * macro_r / (macro_r + macro_p)

    return F1Metrics(
        precision=macro_p,
        recall=macro_r,
        f1=macro_f1
    )


def f1_macro_local(predictions: Iterable[LabeledSample], match_strategy: BaseMatcher) -> F1Metrics:
    raise NotImplementedError()


def calculate_f1(predictions: Iterable[LabeledSample],
                 match_strategy: BaseMatcher,
                 f1_mode: str):
    assert f1_mode in ['micro-local', 'micro-global', 'macro-local', 'macro-global']

    if f1_mode == 'micro-local':
        return f1_micro_local(predictions, match_strategy)

    if f1_mode == 'micro-global':
        return f1_micro_global(predictions, match_strategy)

    if f1_mode == 'macro-local':
        return f1_macro_local(predictions, match_strategy)

    if f1_mode == 'macro-global':
        return f1_macro_global(predictions, match_strategy)


def calculate_confusion_matrix(predictions: Iterable[LabeledSample],
                               label_list: typing.List[str],
                               match_strategy: BoundaryRelationMatcher) -> typing.Tuple[np.ndarray, int, int, int]:
    confusion_matrix = np.zeros(shape=(len(label_list), len(label_list)))
    n_gold_entities = 0
    n_pred_entities = 0
    n_ok_entities = 0
    for p in predictions:
        match_result = match_strategy.match(p)
        matches = match_result.matches
        n_gold_entities += match_result.n_true
        n_pred_entities += match_result.n_pred
        n_ok_entities += match_result.n_ok
        for match in matches:
            confusion_matrix[label_list.index(match[0].tag), label_list.index(match[1].tag)] += 1

    return confusion_matrix, n_gold_entities, n_pred_entities, n_ok_entities
