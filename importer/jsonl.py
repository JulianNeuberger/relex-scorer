import json
import typing

import model
from importer import base


class JsonLinesImporter(base.BaseImporter):
    @staticmethod
    def _token_from_dict(data: typing.Dict[str, typing.Any], default_ner_tag: str = 'O') -> model.Token:
        dependency_relation = data.get('stanza_dependency')
        if dependency_relation is not None:
            dependency_relation = (dependency_relation['head'], dependency_relation['type'])
        return model.Token(
            text=data['text'],
            start_char_index=data['start'],
            stop_char_index=data['stop'],
            ner_tag=data.get('ner', default_ner_tag),
            pos_tag=data.get('stanza_pos'),
            dependency_relation=dependency_relation,
            sentence_id=0
        )

    @staticmethod
    def _entity_from_dict(data: typing.Dict[str, typing.Any], default_ner_tag: str = 'O') -> model.Entity:
        return model.Entity(
            text=data['text'],
            sentence_level_token_indices=data['token_indices'],
            document_level_token_indices=data['token_indices'],
            ner_tag=data.get('ner', default_ner_tag),
            sentence_id=0
        )

    @staticmethod
    def _relation_from_dict(data: typing.Dict[str, typing.Any]) -> model.Relation:
        return model.Relation(
            type=data['type'],
            head=JsonLinesImporter._entity_from_dict(data['head_entity']),
            tail=JsonLinesImporter._entity_from_dict(data['tail_entity'])
        )

    def _sample_from_dict(self, data: typing.Dict[str, typing.Any]) -> model.Sample:
        tokens = [JsonLinesImporter._token_from_dict(d) for d in data['tokens']]
        entities = [JsonLinesImporter._entity_from_dict(d) for d in data['entities']]
        relations = [JsonLinesImporter._relation_from_dict(d) for d in data['relations']]
        self._fix_ner_tags(tokens, entities, relations)
        return model.Sample(
            id=data['id'],
            text=data['text'],
            tokens=tokens,
            entities=entities,
            relations=relations,
        )

    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        if name is None:
            name = file_paths[0]
        assert len(file_paths) == 1
        samples: typing.List[model.Sample] = []
        with open(file_paths[0], 'r', encoding='utf8') as f:
            progress_bar = self._progress_bar(f)
            line = f.readline()
            num_samples = 0
            while line:
                num_samples += 1

                if self._top_n is not None and num_samples > self._top_n:
                    break

                if self._skip and num_samples <= self._skip:
                    self._advance_progress_bar(progress_bar, f)
                    continue

                data = json.loads(line)
                sample = self._sample_from_dict(data)
                samples.append(sample)

                self._advance_progress_bar(progress_bar, f)

                line = f.readline()
        return model.DataSet(samples=samples, name=name)
