import logging
import os
import fire

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
from datasets import Dataset

from nlp_datasets.preprocessing.common_utils import filter_texts_over_tokens_threshold
from nlp_datasets.get_dataset import get_squad_dataset
from nlp_inference.encoder.inference_question_answering import evaluation

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


date = datetime.now().strftime("%H-%M-%S-%d-%m-%Y")
results_dir = Path(__file__).parent.parent.parent / "results" / \
    "question_answering" / "eval" / f"exp_{date}"
print(f"Results directory: {results_dir}")

### START CONFIGURATION PARAMETERS ###
dataset: Dataset = get_squad_dataset()['test']
question_feat: str = "question"
context_feat: str = "context"
answer_feat: str = "answers"
max_tokens_length: int = 512
eval_batch_size: int = 16
### END CONFIGURATION PARAMETERS ###


def start_evaluation(model_name_or_checkpoin_path: str):
    print(f"Preprocessing dataset...")
    dataset_filtered = filter_texts_over_tokens_threshold(dataset=dataset,
                                                 text_feat=context_feat,
                                                 tokenizer_name=model_name_or_checkpoin_path,
                                                 tokens_threshold=max_tokens_length)
    print(f"Dataset filtered: {dataset_filtered}")
    evaluation_metrics: Dict[str, Any] = evaluation(questions=dataset_filtered[question_feat],
                                                    contexts=dataset_filtered[context_feat],
                                                    answers_labels_texts=[el['text'][0] for el in dataset_filtered[answer_feat]],
                                                    answer_starts=[int(el['answer_start'][0]) for el in dataset_filtered[answer_feat]],
                                                    model_name_or_checkpoin_path=model_name_or_checkpoin_path,
                                                    batch_size=eval_batch_size)
    print(evaluation_metrics)
    # write results to file
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / "eval_results.txt", "w") as f:
        f.write(
            f"Model Name or Checkpoint Path:\n{model_name_or_checkpoin_path}\n")
        f.write(f"Evaluation Dataset:\n{dataset}\n")
        f.write(f"Max Tokens Length:\n{max_tokens_length}\n")
        f.write(f"Evaluation Metrics:\n{evaluation_metrics}\n")


if __name__ == "__main__":
    fire.Fire(start_evaluation)
