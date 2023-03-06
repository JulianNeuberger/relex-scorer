import abc
import os
import typing

import stanza.models.common.doc as stanza_model
import tqdm

import model
from importer import util


class BaseImporter(abc.ABC):
    def __init__(self, top_n: int = None, skip: int = 0):
        if top_n is None:
            self._top_n = None
        else:
            self._top_n = top_n + skip
        self._skip = skip

    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        raise NotImplementedError()

    def _fix_ner_tags(self, tokens: typing.List[model.Token],
                      entities: typing.List[model.Entity],
                      relations: typing.List[model.Relation]) -> None:
        for i in range(len(entities)):
            e = entities[i]
            if e.ner_tag == 'O':
                entities[i] = model.Entity(
                    text=e.text,
                    sentence_level_token_indices=e.sentence_level_token_indices,
                    document_level_token_indices=e.document_level_token_indices,
                    ner_tag='UNKNOWN',
                    sentence_id=e.sentence_id
                )

        for i, e in enumerate(entities):
            tags = [tokens[i].ner_tag for i in e.document_level_token_indices]
            if 'O' not in tags:
                continue

            replacement = 'UNKNOWN'
            if e.ner_tag != 'O':
                replacement = e.ner_tag
            tags = [replacement] * len(tags)
            tags = util.plain_ner_tags_to_bioes_tags(tags)

            for token_index, ner_tag in zip(e.document_level_token_indices, tags):
                token = tokens[token_index]
                tokens[token_index] = model.Token(
                    text=token.text,
                    start_char_index=token.start_char_index,
                    stop_char_index=token.stop_char_index,
                    ner_tag=ner_tag,
                    pos_tag=token.pos_tag,
                    dependency_relation=token.dependency_relation,
                    sentence_id=token.sentence_id
                )

        for i in range(len(relations)):
            r = relations[i]
            if r.head.ner_tag == 'O':
                relations[i] = model.Relation(
                    head=model.Entity(
                        text=r.head.text,
                        ner_tag='UNKNOWN',
                        sentence_level_token_indices=r.head.sentence_level_token_indices,
                        document_level_token_indices=r.head.document_level_token_indices,
                        sentence_id=r.head.sentence_id
                    ),
                    tail=r.tail,
                    type=r.type
                )
        for i in range(len(relations)):
            r = relations[i]
            if r.tail.ner_tag == 'O':
                relations[i] = model.Relation(
                    head=r.head,
                    tail=model.Entity(
                        text=r.tail.text,
                        ner_tag='UNKNOWN',
                        sentence_level_token_indices=r.tail.sentence_level_token_indices,
                        document_level_token_indices=r.tail.document_level_token_indices,
                        sentence_id=r.tail.sentence_id
                    ),
                    type=r.type
                )


    def _progress_bar(self, file: typing.IO) -> tqdm.tqdm:
        if self._top_n is None:
            progress_total = os.path.getsize(file.name)
            progress_unit = 'B'
            progress_unit_scale = True
            progress_unit_divisor = 1024
        else:
            progress_total = self._top_n
            progress_unit = 'samples'
            progress_unit_scale = False
            progress_unit_divisor = 1000

        return tqdm.tqdm(total=progress_total,
                         unit=progress_unit,
                         unit_scale=progress_unit_scale,
                         unit_divisor=progress_unit_divisor)

    def _advance_progress_bar(self, progress_bar, f, n_samples: int = 1):
        if self._top_n is None:
            line_size = f.tell() - progress_bar.n
            progress_bar.update(line_size)
        else:
            progress_bar.update(n_samples)

    @staticmethod
    def _build_tokens(doc: stanza_model.Document) -> typing.List[model.Token]:
        stanza_sentence: stanza_model.Sentence
        tokens: typing.List[model.Token] = []
        for sentence_id, stanza_sentence in enumerate(doc.sentences):
            stanza_tokens: typing.List[stanza_model.Token] = stanza_sentence.tokens
            for stanza_token in stanza_tokens:
                stanza_word: stanza_model.Word
                for stanza_word in stanza_token.words:
                    dependency_head = None if stanza_word.head == 0 else stanza_word.head - 1
                    dependency_relation = (dependency_head, stanza_word.deprel)
                    token = model.Token(
                        text=stanza_token.text,
                        ner_tag=stanza_token.ner,
                        pos_tag=stanza_word.xpos,
                        start_char_index=stanza_word.start_char,
                        stop_char_index=stanza_token.end_char,
                        dependency_relation=dependency_relation,
                        sentence_id=sentence_id
                    )
                    tokens.append(token)
        return tokens
