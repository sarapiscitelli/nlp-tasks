import logging
import os

from pathlib import Path
from datetime import datetime
from transformers import TrainingArguments
from datasets import DatasetDict

from nlp_training.encoder.train_text_classification import train_encoder
from nlp_datasets.preprocessing.common_utils import filter_texts_over_tokens_threshold
from nlp_datasets.get_dataset import get_imdb_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

date = datetime.now().strftime("%H-%M-%S-%d-%m-%Y")
results_dir = Path(__file__).parent.parent / "results" / \
    "text_classification" / "train" / f"exp_{date}"
print(f"Results directory: {results_dir}")

### START CONFIGURATION PARAMETERS ###
training_arguments = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=0.0001,
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
    bf16=False
)

base_model_name_or_path: str = "distilbert-base-uncased"
max_tokens_length: int = 512
weighted_loss: bool = True
### END CONFIGURATION PARAMETERS ###

### START DATASET ###
dataset: DatasetDict = get_imdb_dataset()
del dataset['test']
text_feat: str = "text"
num_classes = 2
id2label = {0: "NEGATIVE", 1: "POSITIVE"}

# PREPROCESS DATASET
for split in dataset.keys():
    print(f"Preprocessing {split} split...")
    dataset[split] = filter_texts_over_tokens_threshold(dataset=dataset[split],
                                                        text_feat=text_feat,
                                                        tokenizer_name=base_model_name_or_path,
                                                        tokens_threshold=max_tokens_length)

# START TRAINING
print(f"Dataset:\n{dataset}")
print(f"Training {base_model_name_or_path} model...")
os.makedirs(results_dir, exist_ok=True)
train_encoder(base_model_name_or_path=base_model_name_or_path,
              train_dataset=dataset["train"],
              eval_dataset=dataset["validation"],
              max_tokens_length=max_tokens_length,
              num_classes=num_classes,
              id2label=id2label,
              training_args=training_arguments,
              weighted_loss=weighted_loss)
print("Training succesfully completed")
