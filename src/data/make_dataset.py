from pathlib import Path

from datasets import load_dataset
from datasets import ClassLabel


LABEL_CONVERSION = {'dk': 'Danish',
                    'fo': 'Other',
                    'is': 'Other',
                    'nb': 'Norwegian',
                    'nn': 'Norwegian',
                    'sv': 'Swedish'}


def create_labels(example, id2label_fn):
    example['language_id'] = LABEL_CONVERSION.get(id2label_fn(example['language']))
    return example


def make_dataset(save_path: str=None):
    dataset = load_dataset('strombergnlp/nordic_langid', '10k')
    id2label_fn = dataset['train'].features["language"].int2str
    dataset = dataset.map(create_labels, fn_kwargs={'id2label_fn': id2label_fn})

    dataset = dataset.cast_column('language_id', ClassLabel(
        names=list(set(LABEL_CONVERSION.values()))))
    dataset = dataset.remove_columns('language')
    dataset = dataset.rename_column('language_id', 'language')

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(save_path)
        print(f"Dataset saved in {save_path}")

    return dataset


def main():
    make_dataset()


if __name__ == '__main__':
    main()