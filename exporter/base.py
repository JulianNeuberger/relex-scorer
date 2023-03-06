import abc
import json
import os

import model


class BaseExporter(abc.ABC):
    def save(self, data_set: model.DataSet, file_path: str) -> None:
        raise NotImplementedError()


class LineJsonExporter(BaseExporter):
    @staticmethod
    def _dump_entity_as_dict(entity: model.Entity) -> dict:
        return {
            'text': entity.text,
            'token_indices': entity.sentence_level_token_indices,
            'sentence_id': entity.sentence_id,
            'ner': entity.ner_tag
        }

    @staticmethod
    def _dump_sample_as_dict(sample: model.Sample) -> dict:
        return {
            'id': sample.id,
            'text': sample.text,
            'tokens': [
                {
                    'text': t.text,
                    'start': t.start_char_index,
                    'stop': t.stop_char_index,
                    'ner': t.ner_tag,
                    'stanza_pos': t.pos_tag,
                    'stanza_dependency': {
                        'head': t.dependency_relation[0],
                        'type': t.dependency_relation[1]
                    }
                } for t in sample.tokens
            ],
            'entities': [
                LineJsonExporter._dump_entity_as_dict(e) for e in sample.entities
            ],
            'relations': [
                {
                    'head_entity': LineJsonExporter._dump_entity_as_dict(r.head),
                    'tail_entity': LineJsonExporter._dump_entity_as_dict(r.tail),
                    'type': r.type
                } for r in sample.relations
            ]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            for sample in data_set.samples:
                sample_as_json = json.dumps(LineJsonExporter._dump_sample_as_dict(sample))
                f.write(f'{sample_as_json}\n')
