import numpy as np
import evaluate

from typing import Any, Dict, List, Optional
from functools import partial
from datasets import Dataset

from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    Trainer,
    EvalPrediction
)

from ..custom_modules import WeightedLossClassificationTrainer
from ..callbacks import PrinterCallback

seqeval = evaluate.load("seqeval")


def compute_metrics(label_list: List[str], p: EvalPrediction):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def check_dataset(dataset: Dataset):
    """Dataset must have 'tokens' and 'ner_tags' features
    e.g.  Dataset({
                features: ['id', 'tokens', 'ner_tags'],
                num_rows: 3394
            })
    """
    if "tokens" not in dataset.features:
        raise ValueError("The dataset must have a 'tokens' feature")
    else:
        # check dataset[tokens]
        # e.g. [['@paulwalk', 'It', "'s", 'the', 'view', 'from', ...], ['From', 'Green', 'Newsfeed', ':', 'AHFA', 'extends', ...]
        if not isinstance(dataset['tokens'], list):
            raise ValueError("The dataset[tokens] must be a list of list of tokens")
        else:
            if not isinstance(dataset['tokens'][0], list):
                raise ValueError("The dataset[tokens] must be a list of list of tokens")
    if "ner_tags" not in dataset.features:
        raise ValueError("The dataset must have a 'ner_tags' feature")
    else:
        # check dataset[ner_tags]
        # e.g. [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        if not isinstance(dataset['ner_tags'], list):
            raise ValueError("The dataset[ner_tags] must be a list of list of tags")
        else:
            if not isinstance(dataset['ner_tags'][0], list):
                raise ValueError("The dataset[ner_tags] must be a list of list of tags")
    return True


def train_encoder(
    base_model_name_or_path: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_tokens_length: int,
    num_labels: int,
    id2label: Dict[int, str],
    training_args: TrainingArguments,
    weighted_loss: Optional[bool] = False,
    enable_printer: Optional[bool] = True,
):

    def _tokenize_and_align_labels(examples: Dict[str, List[Any]]):
        """Tokenize data
        Args:
            examples (Dict[str, List[Any]]): examples to tokenize
                e.g. {'text': ['example 1', 'example 2', ...], 
                      'label': [0, 1, ...]}
        """
        tokenized_inputs = tokenizer(examples[dataset_text_field], 
                                     truncation=True, 
                                     max_length=max_tokens_length,
                                     is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[dataset_target_field]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset_text_field = "tokens"
    dataset_target_field = "ner_tags"
    
    # check dataset
    check_dataset(train_dataset)
    check_dataset(eval_dataset)

    # calculate weight loss, if weighted loss is enabled
    loss_weights = None
    if weighted_loss:
        from sklearn.utils.class_weight import compute_class_weight
        loss_weights = compute_class_weight(
                "balanced",
                classes=np.unique(train_dataset[dataset_target_field]),
                y=train_dataset[dataset_target_field])

    # set up model init kwargs (specific for tokenclassification)
    label2id = {v: k for k, v in id2label.items()}
    model_init_kwargs = {'num_labels': num_labels,
                            'id2label': id2label,
                            'label2id': label2id
        }

    # set up callbacks
    callbacks = []
    if enable_printer:
        callbacks.append(PrinterCallback(train_dataset=train_dataset,
                                         eval_dataset=eval_dataset,
                                         max_tokens_length=max_tokens_length,
                                         loss_weights=loss_weights,
                                         num_classes=num_labels,
                                         id2label=id2label,
                                         label2id=label2id))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, 
                                              add_prefix_space=True)
    # tokenize data
    train_dataset = train_dataset.map(_tokenize_and_align_labels, 
                                      batched=True,
                                      remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(_tokenize_and_align_labels, 
                                    batched=True,
                                    remove_columns=eval_dataset.column_names)

    # load data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                       padding="longest",
                                                       max_length=max_tokens_length,
                                                       return_tensors="pt")
    
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name_or_path, **model_init_kwargs
    )
    
    trainer_fn = Trainer if loss_weights is None else WeightedLossClassificationTrainer
    trainer = trainer_fn(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=partial(compute_metrics, list(label2id.keys())),
    )

    # set up loss weights, if weighted loss is enabled
    if loss_weights is not None:
        trainer.set_loss_weights(loss_weights)

    trainer.train()
