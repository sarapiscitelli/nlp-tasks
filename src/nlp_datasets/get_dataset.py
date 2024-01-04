# A collection of standard datasets for NLP tasks.
# Useful for reproduce experiments.

from datasets import load_dataset, DatasetDict


def get_imdb_dataset(seed: int = 12345,
                     validation_size: float = 0.1):
    """
    Get the IMDB movie review dataset.
    Returns:
        DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 22500
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 25000
            })
            validation: Dataset({
                features: ['text', 'label'],
                num_rows: 2500
            })
        })
    """
    imdb: DatasetDict = load_dataset("imdb")
    del imdb['unsupervised']
    # split dataset into train and validation
    imdb['train'], imdb['validation'] = imdb['train'].train_test_split(
        test_size=validation_size, shuffle=True,
        stratify_by_column="label",
        seed=seed).values()

    return imdb


def get_wnut_dataset():
    """
    Get the wnut_17 dataset, for named entity recognition.
    Returns:
        DatasetDict({
            train: Dataset({
                features: ['id', 'tokens', 'ner_tags'],
                num_rows: 3394
            })
            validation: Dataset({
                features: ['id', 'tokens', 'ner_tags'],
                num_rows: 1009
            })
            test: Dataset({
                features: ['id', 'tokens', 'ner_tags'],
                num_rows: 1287
            })
        })
    """
    wnut = load_dataset("wnut_17")
    return wnut


def get_squad_dataset():
    """
    Get the squad dataset, for extractive question answering.
    Returns:
        DatasetDict({
            train: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers'],
                num_rows: 30000
            })
            validation: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers'],
                num_rows: 2000
            })
            test: Dataset({
                features: ['id', 'title', 'context', 'question', 'answers'],
                num_rows: 10570
            })
    """
    squad = load_dataset("squad")
    squad['train'] = squad['train'].select(range(30000))
    squad['test'] = squad['validation']
    squad['validation'] = squad['validation'].select(range(2000))
    return squad


def get_billsum_dataset():
    """
    Get the squad dataset, for generative summarization.
    Returns:
        DatasetDict({
            train: Dataset({
                features: ['text', 'summary', 'title'],
                num_rows: 18949
            })
            test: Dataset({
                features: ['text', 'summary', 'title'],
                num_rows: 3269
            })
            validation: Dataset({
                features: ['text', 'summary', 'title'],
                num_rows: 1237
            })
        })
    """
    billsum = load_dataset("billsum")
    billsum["validation"] = billsum["ca_test"]
    del billsum["ca_test"]
    return billsum


def get_eli5_dataset():
    """Get the eli5 dataset, may be used for causal lm.

    Returns:
        DatasetDict({
            train: Dataset({
                features: ['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls', 'answers_urls', 'text'],
                num_rows: 10000
            })
            validation: Dataset({
                features: ['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls', 'answers_urls', 'text'],
                num_rows: 2281
            })
            test: Dataset({
                features: ['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls', 'answers_urls', 'text'],
                num_rows: 4462
            })
        })
    """
    eli5_train_dataset = load_dataset("eli5", split="train_asks[:10000]")
    eli5_train_dataset = eli5_train_dataset.add_column("text", [el['text'][0] for el in eli5_train_dataset["answers"]])
    eli5_validation_dataset = load_dataset("eli5", split="validation_asks")
    eli5_validation_dataset = eli5_validation_dataset.add_column("text", [el['text'][0] for el in eli5_validation_dataset["answers"]])
    eli5_test_dataset = load_dataset("eli5", split="test_asks")
    eli5_test_dataset = eli5_test_dataset.add_column("text", [el['text'][0] for el in eli5_test_dataset["answers"]])
    dataset = DatasetDict({
        "train": eli5_train_dataset,
        "validation": eli5_validation_dataset,
        "test": eli5_test_dataset
    })
    return dataset
