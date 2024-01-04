import logging

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

def filter_texts_over_tokens_threshold(dataset: Dataset,
                                       text_feat: str,
                                       tokenizer_name: str,
                                       tokens_threshold: int) -> Dataset:
    """Filter out the texts that have more tokens than the threshold.

    Args:
        dataset (Dataset): the dataset to modify
        text_feat (str): the name of the text feature to modify
        tokenizer_name (str): the name of the tokenizer to use to tokenize the texts
        tokens_threshold (int): the threshold to filter out the texts

    Returns:
        Dataset: a new dataset with the modified text feature
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if isinstance(dataset[text_feat][0], list):
        texts = [" ".join(t) for t in dataset[text_feat]]
    else:
        texts = dataset[text_feat]
    token_counts = [len(tokenizer(text).input_ids)
                    for text in tqdm(texts, desc="Tokenizing texts for filtering text over token threshold")]
    idx_to_keep = [idx for idx, t in enumerate(token_counts) if t <= tokens_threshold]
    logger.info(f"Filtering out {len(dataset) - len(idx_to_keep)} texts over the threshold of {tokens_threshold} tokens.")
    return dataset.select(idx_to_keep)
