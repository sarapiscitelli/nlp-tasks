import logging
import os
import fire

from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from datasets import Dataset
from transformers import GenerationConfig

from nlp_datasets.preprocessing.common_utils import filter_texts_over_tokens_threshold
from nlp_datasets.get_dataset import get_eli5_dataset
from nlp_inference.decoder.inference_decoder import evaluation

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


date = datetime.now().strftime("%H-%M-%S-%d-%m-%Y")
results_dir = Path(__file__).parent.parent / "results" / \
    "causal_lm" / "eval" / f"exp_{date}"
print(f"Results directory: {results_dir}")

### START CONFIGURATION PARAMETERS ###
dataset: Dataset = get_eli5_dataset()['test']
text_feat: str = "text"
max_tokens_length: int = 512
eval_batch_size: int = 16
generation_config = GenerationConfig(max_new_tokens=20, 
                                     temperature=0.1, 
                                     repetition_penalty=1.7,
                                     do_sample=True)
### END CONFIGURATION PARAMETERS ###


def create_text_labels_for_evaluation(examples):
    percentage_of_predict_words = 0.2
    texts, labels = [], []
    for t in examples[text_feat]:
        words = t.split(" ")
        words_to_predict = int(len(words) * percentage_of_predict_words)
        texts.append(" ".join(words[:-words_to_predict]))
        labels.append(" ".join(words[-words_to_predict:]))
    return {"text": texts, "labels": labels}

def start_evaluation(model_name_or_checkpoint_path: str = "distilgpt2"):
    print(f"Preprocessing dataset...")
    
    dataset_filtered = filter_texts_over_tokens_threshold(dataset=dataset,
                                                 text_feat=text_feat,
                                                 tokenizer_name=model_name_or_checkpoint_path,
                                                 tokens_threshold=max_tokens_length)
    
    # create text labels for evaluation 
    dataset_filtered = dataset_filtered.map(create_text_labels_for_evaluation, batched=True,
                                            remove_columns=dataset_filtered.features.keys())
        
    print(f"Dataset filtered: {dataset_filtered}")
    evaluation_metrics: Dict[str, Any] = evaluation(texts=dataset_filtered[text_feat],
                                                    text_labels=dataset_filtered["labels"],
                                                    model_name_or_checkpoint_path=model_name_or_checkpoint_path,
                                                    generation_config=generation_config,
                                                    batch_size=eval_batch_size)
    
    print(evaluation_metrics)
    # write results to file
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / "eval_results.txt", "w") as f:
        f.write(
            f"Model Name or Checkpoint Path:\n{model_name_or_checkpoint_path}\n")
        f.write(f"Evaluation Dataset:\n{dataset}\n")
        f.write(f"Max Tokens Length: {max_tokens_length}\n")
        f.write(f"Evaluation Metrics:\n{evaluation_metrics}\n")


if __name__ == "__main__":
    fire.Fire(start_evaluation)
