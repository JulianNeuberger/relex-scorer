import collections
import typing

import numpy as np

import metrics
import model


PredictionAndOriginal = typing.Tuple[metrics.LabeledSample, model.Sample]


def bin_predictions_by(samples: typing.Iterable[PredictionAndOriginal],
                       predicate: typing.Callable[[PredictionAndOriginal], float],
                       num_bins: int = 10
                       ) -> typing.Tuple[typing.Dict[int, typing.Iterable[PredictionAndOriginal]], typing.List[float]]:
    samples = list(samples)
    values = []
    for s in samples:
        try:
            values.append(predicate(s))
        except ValueError:
            continue
    bins = np.linspace(np.min(values), np.max(values) + 1, num_bins + 1)
    bin_indices = np.digitize(values, bins)
    ret = collections.defaultdict(list)

    for i, bin_id in enumerate(bin_indices):
        ret[bin_id].append(samples[i])
        if bin_id == 0:
            print(f'left edge: {bins[0]}, value: {values[i]}')

    assert len(ret) <= num_bins
    assert 0 not in ret
    assert num_bins + 1 not in ret

    return ret, bins.tolist()
