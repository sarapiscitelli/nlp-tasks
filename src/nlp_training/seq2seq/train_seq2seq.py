import evaluate
import numpy as np
import logging

from functools import partial
from typing import Any, Dict, List, Optional
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

from ..utils import LoadDtype
from ..callbacks import PrinterCallback


logger = logging.getLogger(__name__)


metric = evaluate.load("rouge")


def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds: List[str] = [pred.strip() for pred in decoded_preds]
    decoded_labels: List[str]  = [label.strip() for label in decoded_labels]

    # Compute metrics
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}


def check_dataset_format(dataset: Dataset):
    """Dataset should have columns 'text' and 'summary'
    For example:
        test: Dataset({
            features: ['text', 'summary'],
            num_rows: 25000
        })
    """
    if 'text' not in dataset.features:
        raise ValueError(
            "Dataset should have a feature called 'text' for the input text")
    if 'summary' not in dataset.features:
        raise ValueError(
            "Dataset should have a feature called 'summary' for the input label")
    return True


def train(base_model_name_or_path: str,
          train_dataset: Dataset,
          eval_dataset: Dataset,
          max_encoder_length: int,
          max_decoder_length: int,
          training_args: Seq2SeqTrainingArguments,
          quantization_config: Optional[BitsAndBytesConfig] = None,
          load_dtype: str = "auto",
          device_type: Optional[str] = "auto",
          enable_printer: Optional[bool] = True):

    def _tokenize_data(examples: Dict[str, List[Any]]):
        """Tokenize data
        Args:
            examples (Dict[str, List[Any]]): examples to tokenize
                e.g. {'text': ['example 1', 'example 2', ...], 
                      'label': [0, 1, ...]}
        """
        tokenized_inputs = tokenizer(examples[dataset_input_feat],
                            padding="max_length",
                            truncation=True,
                            max_length=max_encoder_length)
        tokenized_targets = tokenizer(examples[dataset_target_feat],
                            padding="max_length",
                            truncation=True,
                            max_length=max_decoder_length)
        # for labels ignore padding in the loss (set -100)
        tokenized_targets["input_ids"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in l]
            for l in tokenized_targets["input_ids"]
        ]
        tokenized_inputs["labels"] = tokenized_targets["input_ids"]
        return tokenized_inputs
        
    dataset_input_feat = "text"
    dataset_target_feat = "summary"

    check_dataset_format(train_dataset)
    check_dataset_format(eval_dataset)

    # set up callbacks
    callbacks = []
    if enable_printer:
        callbacks.append(PrinterCallback(train_dataset=train_dataset,
                                         eval_dataset=eval_dataset,
                                         max_encoder_length=max_encoder_length,
                                         max_decoder_length=max_decoder_length,
                                         quantization_config=quantization_config,
                                         load_dtype=load_dtype,
                                         device_type=device_type
                                         ))
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # tokenize data
    train_dataset = train_dataset.map(_tokenize_data, 
                                      batched=True,
                                      remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(_tokenize_data, 
                                    batched=True,
                                    remove_columns=eval_dataset.column_names)

    if training_args.predict_with_generate == False:
        logger.warning(
            "predict_with_generate=False, this may lead to unexpected behavior for this kind of experiment, verify it's what you want.")

    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path,
                                                  quantization_config=quantization_config,
                                                  device_map=device_type,
                                                  torch_dtype=LoadDtype[load_dtype].value)

    # load data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=partial(compute_metrics, tokenizer),
    )

    trainer.train()
