import logging
import os

from pathlib import Path
from datetime import datetime
from transformers import Seq2SeqTrainingArguments
from datasets import DatasetDict

from nlp_training.seq2seq.train_seq2seq import train
from nlp_datasets.preprocessing.common_utils import filter_texts_over_tokens_threshold
from nlp_datasets.get_dataset import get_billsum_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

date = datetime.now().strftime("%H-%M-%S-%d-%m-%Y")
results_dir = Path(__file__).parent.parent / "results" / \
    "summarization" / "train" / f"exp_{date}"
print(f"Results directory: {results_dir}")

### START CONFIGURATION PARAMETERS ###
training_arguments = Seq2SeqTrainingArguments(
    output_dir=results_dir,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=0.00001,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    eval_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=0.01,
    save_strategy="steps",
    save_steps=0.01,
    logging_strategy="steps",
    logging_steps=1,
    report_to="tensorboard",
    do_train=True,
    do_eval=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    dataloader_drop_last=False,
    fp16=False,
    bf16=False,
    predict_with_generate=True,
    generation_max_length=500
)

base_model_name_or_path: str = "facebook/bart-base"
max_encoder_length: int = 1024
max_decoder_length: int = 1024
### END CONFIGURATION PARAMETERS ###

### START DATASET ###
dataset: DatasetDict = get_billsum_dataset()
del dataset['test']
text_feat: str = "text"
summary_feat: str = "summary"

# PREPROCESS DATASET
for split in dataset.keys():
    print(f"Preprocessing {split} split...")
    dataset[split] = filter_texts_over_tokens_threshold(dataset=dataset[split],
                                                        text_feat=text_feat,
                                                        tokenizer_name=base_model_name_or_path,
                                                        tokens_threshold=max_encoder_length)
    dataset[split] = filter_texts_over_tokens_threshold(dataset=dataset[split],
                                                        text_feat=summary_feat,
                                                        tokenizer_name=base_model_name_or_path,
                                                        tokens_threshold=max_decoder_length)

# START TRAINING
print(f"Dataset:\n{dataset}")
print(f"Training {base_model_name_or_path} model...")
os.makedirs(results_dir, exist_ok=True)
train(base_model_name_or_path=base_model_name_or_path,
              train_dataset=dataset["train"],
              eval_dataset=dataset["validation"],
              max_encoder_length=max_encoder_length,
              max_decoder_length=max_decoder_length,
              training_args=training_arguments)
print("Training succesfully completed")
