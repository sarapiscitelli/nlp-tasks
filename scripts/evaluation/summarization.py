import logging
import os
import fire

from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from datasets import Dataset
from transformers import GenerationConfig

from nlp_datasets.preprocessing.common_utils import filter_texts_over_tokens_threshold
from nlp_datasets.get_dataset import get_billsum_dataset
from nlp_inference.seqseq.inference_seq2seq import evaluation

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


date = datetime.now().strftime("%H-%M-%S-%d-%m-%Y")
results_dir = Path(__file__).parent.parent / "results" / \
    "summarization" / "eval" / f"exp_{date}"
print(f"Results directory: {results_dir}")

### START CONFIGURATION PARAMETERS ###
dataset: Dataset = get_billsum_dataset()['test']
text_feat: str = "text"
summary_feat: str = "summary"
max_encoder_length: int = 1024
max_decoder_length: int = 1024
eval_batch_size: int = 16
generation_config = GenerationConfig(max_new_tokens=100, 
                                     temperature=0.1, 
                                     do_sample=True)
### END CONFIGURATION PARAMETERS ###


def start_evaluation(model_name_or_checkpoin_path: str):
    print(f"Preprocessing dataset...")
    
    dataset_filtered = filter_texts_over_tokens_threshold(dataset=dataset,
                                                 text_feat=text_feat,
                                                 tokenizer_name=model_name_or_checkpoin_path,
                                                 tokens_threshold=max_encoder_length)
    dataset_filtered = filter_texts_over_tokens_threshold(dataset=dataset_filtered,
                                                 text_feat=summary_feat,
                                                 tokenizer_name=model_name_or_checkpoin_path,
                                                 tokens_threshold=max_decoder_length)
    
    dataset_filtered=dataset
    print(f"Dataset filtered: {dataset_filtered}")
    evaluation_metrics: Dict[str, Any] = evaluation(texts=dataset_filtered[text_feat],
                                                    summaries=dataset_filtered[summary_feat],
                                                    model_name_or_checkpoin_path=model_name_or_checkpoin_path,
                                                    generation_config=generation_config,
                                                    batch_size=eval_batch_size)
    print(evaluation_metrics)
    # write results to file
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / "eval_results.txt", "w") as f:
        f.write(
            f"Model Name or Checkpoint Path:\n{model_name_or_checkpoin_path}\n")
        f.write(f"Evaluation Dataset:\n{dataset}\n")
        f.write(f"Max Encoder Length: {max_encoder_length}\n")
        f.write(f"Max Decoder Length: {max_decoder_length}\n")
        f.write(f"Evaluation Metrics:\n{evaluation_metrics}\n")


if __name__ == "__main__":
    fire.Fire(start_evaluation)
