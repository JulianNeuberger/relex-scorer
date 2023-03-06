import random
import typing

import model
from augment import base


class ExampleAugmenter(base.BaseAugmenter):
    def __init__(self, rate: float):
        super().__init__()
        self._rate = rate

    def augment(self, dataset: model.DataSet) -> model.DataSet:
        augmented_dataset = model.DataSet(f'{dataset.name}_augmented', [])
        for sample in dataset.samples:
            augmented_sample = self._replace_tokens_with_example(sample, self._rate)
            augmented_dataset.samples.append(augmented_sample)
        return augmented_dataset

    @staticmethod
    def get_parameters() -> typing.Dict[str, typing.Type]:
        return {
            'rate': float
        }

    @staticmethod
    def _replace_tokens_with_example(sample: model.Sample, replace_chance: float = .1) -> model.Sample:
        assert 0.0 <= replace_chance <= 1.0

        # text all new tokens will have
        new_token_text = 'example'

        # copy data needed for the augmented sample
        augmented_tokens = sample.tokens.copy()
        augmented_sample_text = sample.text
        augmented_relations = [model.Relation(type=r.type, head=r.head, tail=r.tail) for r in sample.relations]

        # we replace tokens, which may shorten/lengthen the sentence length (in chars)
        # this tracks by how much we changed sentence length
        char_offset = 0

        for i, token in enumerate(augmented_tokens):
            if random.random() < replace_chance:
                # will the new token make the sentence longer or shorter?
                sentence_length_change = len(new_token_text) - len(token.text)

                # create the new token, mainly by copying the old one, but adjusting text and start/stop char indices
                offset_adjusted_start_char = token.start_char_index + char_offset
                offset_adjusted_stop_char = token.stop_char_index + char_offset

                # adjust the sample's text by inserting the token
                text_before_old_token = augmented_sample_text[:offset_adjusted_start_char]
                text_after_old_token = augmented_sample_text[offset_adjusted_stop_char:]
                augmented_sample_text = text_before_old_token + new_token_text + text_after_old_token

                example_token = model.Token(text=new_token_text,
                                            start_char_index=offset_adjusted_start_char,
                                            stop_char_index=offset_adjusted_stop_char + sentence_length_change,
                                            sentence_id=token.sentence_id,
                                            ner_tag=token.ner_tag,
                                            pos_tag=token.pos_tag,
                                            dependency_relation=token.dependency_relation)
                augmented_tokens[i] = example_token

                # track the total sentence length change
                char_offset += sentence_length_change

        # regenerate entities texts as they may have been changed by replacing tokens
        augmented_entities = []
        for e in sample.entities:
            head_token = sample.tokens[e.document_level_token_indices[0]]
            tail_token = sample.tokens[e.document_level_token_indices[-1]]

            augmented_entity_text = augmented_sample_text[head_token.start_char_index: tail_token.stop_char_index]

            model.Entity(text=augmented_entity_text,
                         ner_tag=e.ner_tag,
                         sentence_level_token_indices=e.sentence_level_token_indices,
                         document_level_token_indices=e.document_level_token_indices,
                         sentence_id=e.sentence_id)

        # create the augmented sample
        augmented_sample = model.Sample(id=sample.id,
                                        text=augmented_sample_text,
                                        entities=augmented_entities,
                                        relations=augmented_relations,
                                        tokens=augmented_tokens)

        return augmented_sample
