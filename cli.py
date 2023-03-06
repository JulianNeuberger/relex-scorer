import typing

import click

import augment
import classifier
import importer
import model


class AugmentationOption(click.ParamType):
    name = 'augmentation'

    def convert(self, value, param, ctx):
        if isinstance(value, augment.BaseAugmenter):
            return value

        if not isinstance(value, str):
            self.fail(f'Expected value to be of type string, but got {type(value)}.')

        if ':' not in value:
            raw_augmenter_name = value
            raw_augmenter_params = ''
        else:
            try:
                raw_augmenter_name, raw_augmenter_params = value.split(':')
            except ValueError:
                self.fail('Too many separators (":") found, only one or none are allowed.')

        if raw_augmenter_name not in augment.augmenters:
            self.fail(f'Unknown augmenter "{raw_augmenter_name}".')
        augmenter_class = augment.augmenters[raw_augmenter_name]
        augmenter_params = {}
        augmenter_param_types = augmenter_class.get_parameters()

        for param in raw_augmenter_params.split(','):
            param.strip()
            if param == '':
                continue

            param_name, param_value = param.split('=')
            if param_name not in augmenter_param_types:
                self.fail(f'Unknown parameter "{param_name}", only allow {list(augmenter_param_types.keys())}')
            param_type = augmenter_param_types[param_name]

            augmenter_params[param_name] = param_type(param_value)

        return augmenter_class(**augmenter_params)


@click.command()
@click.option('train_data_path', '--train-data', type=click.Path())
@click.option('dev_data_path', '--dev-data', type=click.Path())
@click.option('test_data_path', '--test-data', type=click.Path())
@click.option('classifier_name', '--classifier', '-c', type=click.Choice(classifier.classifiers.keys(), case_sensitive=False))
@click.option('augmentation_steps', '--augmenter', '-a', type=AugmentationOption(), multiple=True)
def main(train_data_path: str, dev_data_path: str, test_data_path: str,
         classifier_name: str, augmentation_steps: typing.List[augment.BaseAugmenter]):
    click.secho(f'Importing data...')
    jsonl_importer = importer.JsonLinesImporter()
    train_data: model.DataSet = jsonl_importer.load(file_paths=[train_data_path])
    dev_data: model.DataSet = jsonl_importer.load(file_paths=[dev_data_path])
    test_data: model.DataSet = jsonl_importer.load(file_paths=[test_data_path])
    click.secho(f'Done!')

    click.secho(f'Agumenting data with {len(augmentation_steps)} steps...')
    augmented_data: model.DataSet = augment.augment_dataset(train_data, augmentation_steps)
    click.secho(f'Done!')

    click.secho(f'Applying classifier "{classifier_name}" to original data...')
    classifier_instance = classifier.classifiers[classifier_name]()
    classifier_instance.fit(train_data, dev_data)
    original_scores = classifier_instance.score(test_data)
    click.secho(f'Scores: '
                f'P={original_scores.precision:06.2%} '
                f'R={original_scores.recall:06.2%} '
                f'F1={original_scores.f1:06.2%}')

    click.secho(f'Applying classifier "{classifier_name}" to augmented data...')
    classifier_instance = classifier.classifiers[classifier_name]()
    classifier_instance.fit(augmented_data, dev_data)
    augmented_scores = classifier_instance.score(test_data)
    click.secho(f'Scores: '
                f'P={augmented_scores.precision:06.2%} '
                f'R={augmented_scores.recall:06.2%} '
                f'F1={augmented_scores.f1:06.2%}')

    f1_improvement_absolute = augmented_scores.f1 - original_scores.f1
    f1_improvement_relative = f1_improvement_absolute / original_scores.f1
    click.secho(f'Augmenting data changed F1 by {f1_improvement_absolute * 100:.2f} percent points, '
                f'or {f1_improvement_relative:.2%} percent.',
                fg='green' if f1_improvement_absolute > 0 else 'red')


if __name__ == '__main__':
    main()
