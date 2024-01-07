import matplotlib.pyplot as plt
import pandas as pd
import logging

from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

logger = logging.getLogger(__name__)


def text_words_statistics(texts: List[str], return_histogram: bool = True) -> Tuple[pd.DataFrame, plt.figure]:
    """Create a histogram counting the words distribution (useful for all NLP tasks)"""
    # Counting the number of words in each text
    word_counts = [len(text.split()) for text in tqdm(texts)]

    # Creating a DataFrame with token statistics
    word_stats_df = pd.DataFrame(word_counts, columns=['Word Counts'])
    word_stats_df = word_stats_df['Word Counts'].agg(
        ['mean', 'min', 'max', 'std'])

    if not return_histogram:
        return word_stats_df, None
    # Plotting the distribution of word counts
    fig, ax = plt.subplots(figsize=(10, 6))
    max_v, min_v = max(word_counts), min(word_counts)
    ax.hist(word_counts, bins=range(min_v, max_v + 2), edgecolor='black')
    ax.set_xlabel('Number of Words')
    ax.set_ylabel('Number of Texts')
    ax.set_title('Distribution of Text Length (in words)')
    plt.xticks(range(min_v, max_v + 1, max(1, int((max_v - min_v) / 20))))
    plt.tight_layout()  # Adjust layout to fit everything neatly
    return word_stats_df, fig


def text_tokens_statistics(texts: List[str], tokenizer_name: str, return_histogram: bool = True) -> Tuple[pd.DataFrame, plt.figure]:
    """Create a histogram counting the tokens distribution (useful for all NLP tasks)"""
    # Counting the number of tokens in each text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_counts = [len(tokenizer.tokenize(text)) for text in tqdm(texts)]

    # Creating a DataFrame with token statistics
    token_stats_df = pd.DataFrame(token_counts, columns=['Token Counts'])
    token_stats_df = token_stats_df['Token Counts'].agg(
        ['mean', 'min', 'max', 'std'])
    if not return_histogram:
        return token_stats_df, None
    # Plotting the distribution of token counts
    fig, ax = plt.subplots(figsize=(10, 6))
    max_v, min_v = max(token_counts), min(token_counts)
    ax.hist(token_counts, bins=range(min_v, max_v + 2), edgecolor='black')
    ax.set_xlabel('Number of Tokens')
    ax.set_ylabel('Number of Texts')
    ax.set_title(f'Distribution of Token Counts - {tokenizer_name}')
    plt.xticks(range(min_v, max_v + 1, max(1, int((max_v - min_v) / 20))))
    plt.tight_layout()  # Adjust layout to fit everything neatly

    return token_stats_df, fig


def labels_statistics(labels: List[str], return_histogram: bool = True) -> Tuple[pd.DataFrame, plt.figure]:
    """Create a histogram counting the label distribution (useful for classification tasks)"""
    # Creating a DataFrame with token statistics
    label_stats_df = pd.DataFrame(labels, columns=['Label Counts'])
    label_stats_df = label_stats_df['Label Counts'].agg(
        ['mean', 'min', 'max', 'std'])
    if not return_histogram:
        return label_stats_df, None
    # Plotting the distribution of token counts
    fig, ax = plt.subplots(figsize=(10, 6))
    max_v, min_v = max(labels), min(labels)
    ax.hist(labels, bins=range(min_v, max_v + 2), edgecolor='black')
    ax.set_xlabel('Number of Labels')
    ax.set_ylabel('Label Types')
    ax.set_title(f'Distribution of Label Counts')
    plt.xticks(range(min_v, max_v + 1))
    plt.tight_layout()  # Adjust layout to fit everything neatly

    return label_stats_df, fig
