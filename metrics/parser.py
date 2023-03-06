import json
import typing
from dataclasses import dataclass
from typing import Iterable, Optional

import tqdm

import model


@dataclass
class LabeledEntity:
    start: int
    end: int

    tag: Optional[str]

    def __eq__(self, o: 'LabeledEntity') -> bool:
        if type(o) is not LabeledEntity:
            return False
        return o.start == self.start and o.end == self.end and o.tag == self.tag

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.tag))


@dataclass
class LabeledRelation:
    head: LabeledEntity
    tail: LabeledEntity

    tag: str

    def to_tuple(self, use_ner: bool = False):
        if use_ner:
            return self.head.start, self.head.end, self.head.tag, self.tail.start, self.tail.end, self.tail.tag, self.tag
        return self.head.start, self.head.end, self.tail.start, self.tail.end, self.tag

    def __eq__(self, o: 'LabeledRelation') -> bool:
        if type(o) is not LabeledRelation:
            return False
        return o.head == self.head and o.tail == self.tail and o.tag == self.tag

    def __hash__(self) -> int:
        return hash((self.head, self.tail, self.tag))


@dataclass
class LabeledSample:
    sample_id: str
    prediction: typing.List[LabeledRelation]
    labels: typing.List[LabeledRelation]


class BaseParser:
    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        raise NotImplementedError()

    @staticmethod
    def _build_sample_index(dataset: model.DataSet) -> typing.Dict[str, model.Sample]:
        ret = {}
        for s in dataset.samples:
            ret[str(s.id)] = s
        return ret

    @staticmethod
    def insert_missing_samples(parsed_samples: typing.List[LabeledSample],
                               original_dataset: model.DataSet) -> None:
        parsed_ids = sorted([str(s.sample_id) for s in parsed_samples])
        penalized_ids = []
        for s in original_dataset.samples:
            sample_id = str(s.id)
            if sample_id not in parsed_ids:
                penalized_ids.append(sample_id)
                parsed_samples.append(LabeledSample(
                    sample_id=sample_id,
                    prediction=[],
                    labels=[
                        LabeledRelation(
                            head=LabeledEntity(start=min(r.head.token_indices),
                                               end=max(r.head.token_indices),
                                               tag=r.head.ner_tag),
                            tail=LabeledEntity(start=min(r.tail.token_indices),
                                               end=max(r.tail.token_indices),
                                               tag=r.tail.ner_tag),
                            tag=r.type
                        ) for r in s.relations
                    ]
                ))
        if len(penalized_ids) > 0:
            print(f'Penalized approach by adding {len(penalized_ids)} empty predictions for omitted samples.')


class PFNCustomParser(BaseParser):
    def _parse_pred_entity(self, data: typing.Dict) -> LabeledEntity:
        return LabeledEntity(
            start=data['start'],
            end=data['stop'],
            tag=None
        )

    def _parse_pred_relation(self, data: typing.Dict) -> LabeledRelation:
        return LabeledRelation(
            head=self._parse_pred_entity(data['head_entity']),
            tail=self._parse_pred_entity(data['tail_entity']),
            tag=data['type']
        )

    def _parse_prediction(self, data: typing.Dict) -> typing.Tuple[str, typing.List[LabeledRelation]]:
        return data['id'], [self._parse_pred_relation(r) for r in data['relations']]

    def _parse_prediction_chunk(self, data: typing.List[typing.Dict]) -> typing.Dict[str, typing.List[LabeledRelation]]:
        ret = {}
        for prediction in data:
            sample_id, sample = self._parse_prediction(prediction)
            ret[sample_id] = sample
        return ret

    def _parse_token(self, data: typing.Dict) -> typing.Tuple[int, int]:
        return data['start'], data['stop']

    def _parse_label_entity(self, data: typing.Dict, tokens: typing.List[typing.Tuple[int, int]]) -> LabeledEntity:
        return LabeledEntity(
            start=tokens[min(data['token_indices'])][0],
            end=tokens[max(data['token_indices'])][1] + 1,
            tag=data['ner']
        )

    def _parse_label_relation(self, data: typing.Dict, tokens: typing.List[typing.Tuple[int, int]]) -> LabeledRelation:
        return LabeledRelation(
            head=self._parse_label_entity(data['head_entity'], tokens),
            tail=self._parse_label_entity(data['tail_entity'], tokens),
            tag=data['type']
        )

    def _parse_label(self, data: typing.Dict) -> typing.Tuple[str, typing.List[LabeledRelation]]:
        tokens = [self._parse_token(t) for t in data['tokens']]
        return data['id'], [self._parse_label_relation(r, tokens) for r in data['relations']]

    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        file_paths = list(file_paths)
        assert len(file_paths) == 2
        label_file_path, prediction_file_path = file_paths
        with open(prediction_file_path, 'r', encoding='utf8') as f:
            data = json.load(f)
            predictions = {}
            for chunk in data:
                predictions.update(self._parse_prediction_chunk(chunk))
        with open(label_file_path, 'r', encoding='utf8') as f:
            labels = {}
            for line in f:
                data = json.loads(line)
                sample_id, relations = self._parse_label(data)
                labels[sample_id] = relations

        assert labels.keys() == predictions.keys()

        ret = []
        for sample_id in labels.keys():
            s = LabeledSample(
                sample_id=sample_id,
                labels=labels[sample_id],
                prediction=predictions[sample_id]
            )
            ret.append(s)
        return ret


class JointERParser(BaseParser):
    @staticmethod
    def _parse_tagged(raw: Iterable, text: str):
        ret = []
        for r in raw:
            head_token, r_tag, tail_token = r
            h_start = text.find(head_token.lower())
            h_end = h_start + len(head_token)
            h_tag = None

            t_start = text.find(tail_token.lower())
            t_end = t_start + len(tail_token)
            t_tag = None

            ret.append(LabeledRelation(
                head=LabeledEntity(h_start, h_end, h_tag),
                tail=LabeledEntity(t_start, t_end, t_tag),
                tag=r_tag
            ))
        return ret

    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        file_paths = list(file_paths)
        assert len(file_paths) == 1

        with open(file_paths[0], 'r', encoding='utf8') as f:
            samples = json.load(f)

        ret = []

        for sample in samples:
            text: str = sample['text'].lower()
            pred = sample['predict']
            true = sample['truth']

            predicted_rels = JointERParser._parse_tagged(pred, text)
            true_rels = JointERParser._parse_tagged(true, text)

            ret.append(LabeledSample(
                sample_id=sample['id'],
                prediction=predicted_rels,
                labels=true_rels
            ))

        return ret


class TwoAreBetterThanOneParser(BaseParser):
    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        file_paths = list(file_paths)
        assert len(file_paths) == 1

        with open(file_paths[0], 'r', encoding='utf8') as f:
            data = json.load(f)
        batched_relations = data['relations']

        samples = []
        for b in batched_relations:
            serialized_samples = zip(b['true'], b['pred'], b['sample_id'])
            for serialized_sample in serialized_samples:
                trues = []
                preds = []
                for true in serialized_sample[0]:
                    trues.append(TwoAreBetterThanOneParser._parse_single(true))
                for pred in serialized_sample[1]:
                    preds.append(TwoAreBetterThanOneParser._parse_single(pred))
                sample_id = serialized_sample[2]
                samples.append(LabeledSample(prediction=preds, labels=trues, sample_id=sample_id))
        return samples

    @staticmethod
    def _parse_single(sample) -> LabeledRelation:
        if len(sample) == 7:
            # labeled relation with ner tagged entities
            h_start, h_end, h_tag, t_start, t_end, t_tag, r_tag = sample
        elif len(sample) == 5:
            # labeled relation w/out ner tagged entities
            h_start, h_end, t_start, t_end, r_tag = sample
            h_tag = None
            t_tag = None
        else:
            raise ValueError(f'Unknown prediction type {sample}')
        return LabeledRelation(
            head=LabeledEntity(h_start, h_end, h_tag),
            tail=LabeledEntity(t_start, t_end, t_tag),
            tag=r_tag
        )


class CasRelParser(BaseParser):
    @staticmethod
    def _parse_entity(text: str, entity_text: str) -> LabeledEntity:
        start = text.find(entity_text)
        stop = start + len(entity_text) + 1
        return LabeledEntity(start=start, end=stop, tag=None)

    @staticmethod
    def _parse_relation(text: str, relation_data: typing.Dict) -> LabeledRelation:
        return LabeledRelation(
            head=CasRelParser._parse_entity(text, relation_data['subject']),
            tail=CasRelParser._parse_entity(text, relation_data['object']),
            tag=relation_data['relation']
        )

    @staticmethod
    def _parse_sample(serialized_sample: str) -> LabeledSample:
        data = json.loads(serialized_sample)
        text: str = data['text']

        preds = []
        labels = []

        for rel in data['triple_list_gold']:
            labels.append(CasRelParser._parse_relation(text, rel))

        for rel in data['triple_list_pred']:
            preds.append(CasRelParser._parse_relation(text, rel))

        return LabeledSample(
            sample_id=data['id'],
            prediction=preds,
            labels=labels
        )

    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        file_paths = list(file_paths)
        assert len(file_paths) == 1

        labeled_samples = []
        with open(file_paths[0], 'r', encoding='utf8') as f:
            serialized_sample = ''
            line: str
            for line in tqdm.tqdm(f):
                serialized_sample += line.strip()
                if line.rstrip() == '}':
                    labeled_samples.append(self._parse_sample(serialized_sample))
                    serialized_sample = ''
        return labeled_samples


class RSANParser(BaseParser):
    @staticmethod
    def _parse_entity(data):
        return LabeledEntity(start=data[1], end=data[2], tag=None)

    @staticmethod
    def _parse_relation(data):
        e1, e2, r_type = data
        head = e1 if e1[0] == 'H' else e2
        tail = e1 if e1[0] == 'T' else e2
        return LabeledRelation(
            head=RSANParser._parse_entity(head),
            tail=RSANParser._parse_entity(tail),
            tag=str(r_type)
        )

    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        file_paths = list(file_paths)
        assert len(file_paths) == 1
        with open(file_paths[0], 'r', encoding='utf8') as f:
            results = json.load(f)
        predictions = results['predictions']
        labels = results['target']
        ids = results['ids']

        labeled_samples = []
        for true, pred, sample_id in zip(labels, predictions, ids):
            true_relations = [RSANParser._parse_relation(r) for r in true]
            pred_relations = [RSANParser._parse_relation(r) for r in pred]
            labeled_samples.append(LabeledSample(
                prediction=pred_relations,
                labels=true_relations,
                sample_id=sample_id
            ))
        return labeled_samples


class SpertParser(BaseParser):
    def _parse_predicted_relation(self, data: typing.Dict, entities: typing.List[LabeledEntity]) -> LabeledRelation:
        return LabeledRelation(
            head=entities[data['head']],
            tail=entities[data['tail']],
            tag=data['type']
        )

    def _parse_label_relation(self, data: typing.Dict) -> LabeledRelation:
        return LabeledRelation(
            head=self._parse_label_entity(data['head_entity']),
            tail=self._parse_label_entity(data['tail_entity']),
            tag=data['type']
        )

    def _parse_label_entity(self, data: typing.Dict) -> LabeledEntity:
        return LabeledEntity(
            start=min(data['token_indices']),
            end=max(data['token_indices']) + 1,
            tag=data['ner']
        )

    def _parse_predicted_entity(self, data: typing.Dict) -> LabeledEntity:
        return LabeledEntity(
            start=data['start'],
            end=data['end'],
            tag=data['type']
        )

    def _parse_sample(self, label_data: typing.Dict, pred_data: typing.Dict) -> LabeledSample:
        label_relations = [self._parse_label_relation(relation_data) for relation_data in label_data['relations']]

        pred_entities = [self._parse_predicted_entity(entity_data) for entity_data in pred_data['entities']]
        pred_relations = [self._parse_predicted_relation(relation_data, pred_entities) for relation_data in
                          pred_data['relations']]

        return LabeledSample(
            sample_id=label_data['id'],
            prediction=pred_relations,
            labels=label_relations
        )

    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        file_paths = list(file_paths)
        assert len(file_paths) == 2

        labels_file_path, predictions_file_path = file_paths

        with open(labels_file_path, 'r', encoding='utf8') as f:
            labels = [json.loads(line) for line in f]
        with open(predictions_file_path, 'r', encoding='utf8') as f:
            preds = json.load(f)

        if len(labels) != len(preds):
            print(len(labels))
            print(len(preds))
            raise AssertionError()

        return [
            self._parse_sample(label_data, pred_data)
            for label_data, pred_data
            in zip(labels, preds)
        ]


class MareParser(BaseParser):
    def _parse_token(self, data: typing.Dict) -> typing.Tuple[int, int]:
        return data['start'], data['stop']

    def _parse_label_entity(self, data: typing.Dict, tokens: typing.List[typing.Tuple[int, int]]) -> LabeledEntity:
        return LabeledEntity(
            start=tokens[min(data['token_indices'])][0],
            end=tokens[max(data['token_indices'])][1],
            tag=data['ner']
        )

    def _parse_label_relation(self, data: typing.Dict, tokens: typing.List[typing.Tuple[int, int]]) -> LabeledRelation:
        return LabeledRelation(
            head=self._parse_label_entity(data['head_entity'], tokens),
            tail=self._parse_label_entity(data['tail_entity'], tokens),
            tag=data['type']
        )

    def _parse_label(self, data: typing.Dict) -> typing.Tuple[str, typing.List[LabeledRelation]]:
        tokens = [self._parse_token(t) for t in data['tokens']]
        return str(data['id']), [self._parse_label_relation(r, tokens) for r in data['relations']]

    def _parse_prediction_entity(self, data: typing.Dict) -> LabeledEntity:
        return LabeledEntity(
            start=data['start'],
            end=data['stop'],
            tag=None
        )

    def _parse_prediction_relation(self, data: typing.Dict) -> LabeledRelation:
        return LabeledRelation(
            head=self._parse_prediction_entity(data['head_entity']),
            tail=self._parse_prediction_entity(data['tail_entity']),
            tag=data['type']
        )

    def parse(self, file_paths: Iterable[str], original_dataset: model.DataSet) -> Iterable[LabeledSample]:
        sample_index = self._build_sample_index(original_dataset)
        file_paths = list(file_paths)
        assert len(file_paths) == 2
        labels_file_path, predictions_file_path = file_paths
        with open(labels_file_path, 'r', encoding='utf8') as f:
            labels = {}
            for line in f:
                data = json.loads(line)
                sample_id, relations = self._parse_label(data)
                labels[sample_id] = relations
        with open(predictions_file_path, 'r', encoding='utf8') as f:
            predictions = {}
            for line in f:
                data = json.loads(line)
                sample_id = str(data['id'])
                relations = [self._parse_prediction_relation(r) for r in data['relations']]
                predictions[sample_id] = relations

        assert labels.keys() == predictions.keys()

        ret = []
        for sample_id in labels.keys():
            s = LabeledSample(
                sample_id=sample_id,
                labels=labels[sample_id],
                prediction=predictions[sample_id]
            )
            ret.append(s)
        return ret
