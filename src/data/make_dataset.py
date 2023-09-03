from datasets import load_dataset
from datasets import ClassLabel


LABEL_CONVERSION = {'dk': 'Danish',
                    'fo': 'Other',
                    'is': 'Other',
                    'nb': 'Norwegian',
                    'nn': 'Norwegian',
                    'sv': 'Swedish'}


#def id2label_fn(dataset):
#    return dataset['train'].features["language"].int2str


def create_labels(example, id2label_fn):
    example['aaa'] = LABEL_CONVERSION.get(id2label_fn(example['language']))
    return example


def make_dataset():
    dataset = load_dataset('strombergnlp/nordic_langid', '10k')
    id2label_fn = dataset['train'].features["language"].int2str
    aa = dataset.map(create_labels, fn_kwargs={'id2label_fn': id2label_fn})

    aa = aa.cast_column('aaa', ClassLabel(names=list(set(LABEL_CONVERSION.values()))))
    aa = aa.remove_columns('language')
    aa = aa.rename_column('aaa', 'language')

    print('kk')


def main():
    make_dataset()


if __name__ == '__main__':
    main()