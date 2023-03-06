import typing


def plain_ner_tag_from_bioes(bioes_ner_tag: str) -> str:
    if bioes_ner_tag == 'O':
        return bioes_ner_tag
    return bioes_ner_tag.split('-', maxsplit=1)[1]


def plain_ner_tags_to_bioes_tags(ner_tags: typing.List[str]) -> typing.List[str]:
    def bioes_tag(plain_tag: str, index: int, length: int) -> str:
        if plain_tag == 'O':
            return plain_tag

        if length == 1:
            return f'S-{plain_tag}'

        if index == 0:
            return f'B-{plain_tag}'
        if index == length - 1:
            return f'E-{plain_tag}'

        return f'I-{plain_tag}'

    num_ner_tags = len(ner_tags)
    assert num_ner_tags > 0
    return [bioes_tag(t, i, num_ner_tags) for i, t in enumerate(ner_tags)]