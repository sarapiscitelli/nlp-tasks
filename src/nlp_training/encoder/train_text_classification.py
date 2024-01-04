import numpy as np
import evaluate

from typing import Any, Dict, List, Optional
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    Trainer,
    EvalPrediction
)

from ..callbacks import PrinterCallback
from ..custom_modules import WeightedLossClassificationTrainer

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


def check_dataset_format(dataset: Dataset):
    """Dataset should have columns 'text' and 'label'
    For example:
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 25000
        })
    """
    if 'text' not in dataset.features:
        raise ValueError(
            "Dataset should have a feature called 'text' for the input text")
    if 'label' not in dataset.features:
        raise ValueError(
            "Dataset should have a feature called 'label' for the input label")
    return True

def train_encoder(
    base_model_name_or_path: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_tokens_length: int,
    training_args: TrainingArguments,
    num_classes: int,
    id2label: Dict[int, str],
    weighted_loss: Optional[bool] = False,
    enable_printer: Optional[bool] = True,
):

    def _tokenize_data(examples: Dict[str, List[Any]]):
        """Tokenize data
        Args:
            examples (Dict[str, List[Any]]): examples to tokenize
                e.g. {'text': ['example 1', 'example 2', ...], 
                      'label': [0, 1, ...]}
        """
        return tokenizer(
            examples[dataset_text_field],
            truncation=True,
            max_length=max_tokens_length,
        )


    dataset_text_field = "text"
    dataset_target_field = "label"
    
    check_dataset_format(train_dataset)
    check_dataset_format(eval_dataset)

    # calculate weight loss, if weighted loss is enabled
    loss_weights = None
    if weighted_loss:
        from sklearn.utils.class_weight import compute_class_weight
        loss_weights = compute_class_weight(
                "balanced",
                classes=np.unique(train_dataset["label"]),
                y=train_dataset["label"])

    # set up model init kwargs (specific for classification)
    label2id = {v: k for k, v in id2label.items()}
    model_init_kwargs = {"num_labels": num_classes,
                         "id2label": id2label,
                         "label2id": label2id
                         }

    # set up callbacks
    callbacks = []
    if enable_printer:
        callbacks.append(PrinterCallback(train_dataset=train_dataset,
                                         eval_dataset=eval_dataset,
                                         max_tokens_length=max_tokens_length,
                                         loss_weights=loss_weights,
                                         num_classes=num_classes,
                                         id2label=id2label,
                                         label2id=label2id))
        
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    # tokenize data
    train_dataset = train_dataset.map(_tokenize_data,
                                      batched=True)
    eval_dataset = eval_dataset.map(_tokenize_data,
                                    batched=True)

    # load data collator
    data_collator = DataCollatorWithPadding(tokenizer,
                                            padding="longest",
                                            return_tensors="pt")

    model = AutoModelForSequenceClassification.from_pretrained(
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
        compute_metrics=compute_metrics,
    )

    # set up loss weights, if weighted loss is enabled
    if loss_weights is not None:
        trainer.set_loss_weights(loss_weights)

    trainer.train()

    trainer.predict(test_dataset=eval_dataset)
    