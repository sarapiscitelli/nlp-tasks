import logging
import numpy as np
import evaluate

from typing import List, Optional
from functools import partial
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments)

from ..utils import LoadDtype, build_peft_config
from ..callbacks import PrinterCallback


logger = logging.getLogger(__name__)

metric = evaluate.load("rouge")

def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=-1)
    # shift predictions and labels to the right for evaluation
    preds = preds[..., :-1]
    labels = labels[..., 1:]
    # Replace the tokens that correspond to labels padded with -100
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds: List[str] = [pred.strip() for pred in decoded_preds]
    decoded_labels: List[str] = [label.strip() for label in decoded_labels]

    # Compute metrics
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}

def check_dataset_format(dataset: Dataset):
    """_summary_
    For example:
        train: Dataset({
            features: ['text'],
            num_rows: 25000
        })
    """
    if 'text' not in dataset.features:
        raise ValueError(
            "Dataset should have a feature called 'text' for the input text")
    else:
        # check if it is a list of strings
        # dataset['text']: List[str] = [txt_1, txt_2]
        if not isinstance(dataset['text'], list):
            raise ValueError(f"Dataset['text'] should be a list of strings, but got {type(dataset['text'])}")
    return True


def train(base_model_name_or_path: str,
          train_dataset: Dataset,
          eval_dataset: Dataset,
          training_args: TrainingArguments,
          max_tokens_length: int,
          end_of_instruction_template: Optional[str] = None,
          peft_config: Optional[PeftConfig] = None,
          quantization_config: Optional[BitsAndBytesConfig] = None,
          load_dtype: str = "auto",
          device_type: Optional[str] = "auto",
          enable_printer: Optional[bool] = True):

    dataset_input_feat = "text"
    check_dataset_format(train_dataset)
    check_dataset_format(eval_dataset)

    # set up callbacks
    callbacks = []
    if enable_printer:
        callbacks.append(PrinterCallback(train_dataset=train_dataset,
                                         eval_dataset=eval_dataset,
                                         max_tokens_length=max_tokens_length,
                                         peft_config=peft_config,
                                         end_of_instruction_template=end_of_instruction_template,
                                         quantization_config=quantization_config,
                                         load_dtype=load_dtype,
                                         device_type=device_type))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    #tokenizer.add_bos_token = False

    # load data collator
    if end_of_instruction_template is None:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
    else:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=end_of_instruction_template, tokenizer=tokenizer, mlm=False
        )

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_type,
        torch_dtype=LoadDtype[load_dtype].value
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    if peft_config is not None:
        peft_config = build_peft_config(peft_config, base_model)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
        dataset_text_field=dataset_input_feat,
        max_seq_length=max_tokens_length,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        compute_metrics=partial(compute_metrics, tokenizer),
        packing=False
    )

    trainer.train()
    