import json
import os
import typing

import model
from exporter import base


class DocRedExporter(base.BaseExporter):
    @staticmethod
    def _dump_entity_as_dict(entity: model.Entity,
                             entity_to_id: typing.Dict[model.Entity, int]) -> typing.Dict:
        entity_to_id[entity] = len(entity_to_id)
        return {
            'pos': (min(entity.sentence_level_token_indices), max(entity.sentence_level_token_indices) + 1),
            'type': entity.ner_tag,
            'sent_id': entity.sentence_id,
            'name': entity.text
        }

    @staticmethod
    def _dump_relation_as_dict(relation: model.Relation,
                               sample: model.Sample) -> typing.Dict:
        return {
            'r': relation.type,
            'h': sample.entities.index(relation.head),
            't': sample.entities.index(relation.tail),
            'evidence': list({relation.head.sentence_id, relation.tail.sentence_id})
        }

    @staticmethod
    def _dump_tokens_as_nested_list(tokens: typing.List[model.Token]) -> typing.List[typing.List[str]]:
        ret = []
        last_sentence_id = None
        for token in tokens:
            if token.sentence_id != last_sentence_id:
                last_sentence_id = token.sentence_id
                ret.append([])
            ret[-1].append(token.text)
        return ret

    def _dump_sample_as_dict(self, sample: model.Sample) -> typing.Dict:
        entity_to_id = {}
        serialized_entities = [self._dump_entity_as_dict(e, entity_to_id) for e in sample.entities]

        return {
            'vertexSet': [[s] for s in serialized_entities],
            'labels': [self._dump_relation_as_dict(r, sample) for r in sample.relations],
            'title': sample.id,
            'sents': DocRedExporter._dump_tokens_as_nested_list(sample.tokens)
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        serialized_samples = []
        for s in data_set.samples:
            serialized_samples.append(self._dump_sample_as_dict(s))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(serialized_samples, f)
