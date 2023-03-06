from dataclasses import dataclass
from typing import Iterable, Set

# TODO: abstract
# h_from_idx, h_to_idx, h_ne_tag, t_from_idx, t_to_idx, t_ne_tag, re_type
import typing

from metrics.parser import LabeledRelation, LabeledSample, LabeledEntity


@dataclass
class SpanMatches:
    prediction: LabeledSample

    n_true: float
    n_pred: float

    n_ok: float

    matches: typing.List[typing.Tuple[typing.Union[LabeledRelation, LabeledEntity], typing.Union[LabeledRelation, LabeledEntity]]]

    def __str__(self) -> str:
        return f'SpanMatches[#gold:{self.n_true}|#pred:{self.n_pred}|#ok:{self.n_ok}]'


class BaseMatcher:
    def match(self, prediction: LabeledSample) -> SpanMatches:
        raise NotImplementedError()


class ExactRelationMatcher(BaseMatcher):
    @staticmethod
    def _to_set(labeled: Iterable[LabeledRelation]) -> Set:
        return set([
            labeled_relation.to_tuple()
            for labeled_relation in labeled
        ])

    def match(self, prediction: LabeledSample) -> SpanMatches:
        t = ExactRelationMatcher._to_set(prediction.labels)
        p = ExactRelationMatcher._to_set(prediction.prediction)

        # intersect the two sets
        hits = t & p

        # TODO: calculate matches, not needed for now
        return SpanMatches(prediction, len(t), len(p), len(hits), [])


class BoundaryRelationMatcher(BaseMatcher):
    def match(self, prediction: LabeledSample) -> SpanMatches:
        matches: typing.List[typing.Tuple[LabeledRelation, LabeledRelation]] = []

        labels = set(prediction.labels)
        preds = set(prediction.prediction)

        for t in labels:
            for p in prediction.prediction:
                head_boundaries_match = t.head.start == p.head.start and t.head.end == p.head.end
                tail_boundaries_match = t.tail.start == p.tail.start and t.tail.end == p.tail.end
                if head_boundaries_match and tail_boundaries_match:
                    matches.append((t, p))
                    break
        return SpanMatches(prediction,
                           n_true=len(labels),
                           n_pred=len(preds),
                           n_ok=len(matches), matches=matches)


class EntityMatcher(BaseMatcher):
    def match(self, prediction: LabeledSample) -> SpanMatches:
        matches: typing.List[typing.Tuple[LabeledEntity, LabeledEntity]] = []

        labels = set([x.head for x in prediction.labels]).union([x.tail for x in prediction.labels])
        preds = set([x.head for x in prediction.prediction]).union([x.tail for x in prediction.prediction])

        for t in labels:
            for p in preds:
                if t.start == p.start and t.end == p.end:
                    matches.append((t, p))
                    break

        num_true = len(labels)
        num_pred = len(preds)

        assert len(matches) <= num_true and len(matches) <= num_pred, \
            f'Found {len(matches)} matches, but only have {num_true} gold and {num_pred} predictions. \n' \
            f'Matches: {matches}\n' \
            f'True   : {labels}\n' \
            f'Preds  : {preds}'

        return SpanMatches(prediction,
                           n_true=num_true, n_pred=num_pred,
                           n_ok=len(matches), matches=matches)
