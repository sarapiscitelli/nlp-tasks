import enum
import torch

from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


class InferenceModel:

    def __init__(self, model_name_or_checkpoin_path: str,
                 tokenizer_name: Optional[str] = None,
                 device_type: Optional[str] = None) -> None:
        if tokenizer_name is None:
            tokenizer_name = model_name_or_checkpoin_path
        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = AutoModelForTokenClassification.from_pretrained(model_name_or_checkpoin_path, device_map=device_type)
        if 'id2label' not in self._model.config.__dict__:
            raise ValueError(f"The model must be a sequence classification model, 'id2label' is not found in the model config = {self.model.config.__dict__}")
        self._id2label = self._model.config.id2label
        self._model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_checkpoin_path,
                                                       add_prefix_space=True)

    @property
    def id2label(self):
        return self._id2label

    @property
    def model(self):
        return self._model


    def inference(self, texts: List[List[str]]) -> List[List[str]]:
        """_summary_

        Args:
            texts (List[str]): _description_

        Returns:
            Tuple[List[List[str]], List[List[str]]]: _description_
                e.g. ([["t_1(1)", "t_1(2)", "t_1(3)"], ["t_2(1)", "t_2(2)", "t_2(3)", "t_2(4)"]], 
                        [["O", "O", "O"], ["O", "O", "B-PER", "O"]])
        """
        inputs = self.tokenizer(texts,
                                truncation=True,
                                padding="longest",
                                is_split_into_words=True,
                                return_offsets_mapping=True,  # To get the mapping
                                return_tensors="pt").to(self._model.device)
        offsets_mapping = inputs.pop("offset_mapping")

        with torch.no_grad():
            logits = self._model(**inputs).logits

        predicted_class_id = logits.argmax(dim=-1).tolist()
        predicted_classes = []

        for i in range(0, len(texts)):
            sentence_classes = []
            for j, offsets in enumerate(offsets_mapping[i]):
                # Skip special tokens
                if offsets[0] == 0 and offsets[1] == 0:
                    continue
                if offsets[0] == 0 and offsets[1] > 0:
                    # begin of a new word
                    sentence_classes.append(self.id2label[predicted_class_id[i][j]])
                    continue
                if offsets[0] > 0 and offsets[1] > 0:
                    # continue of a word
                    continue
            predicted_classes.append(sentence_classes)

        return predicted_classes
              

def evaluation(tokens: List[List[str]],
               labels_tokens: List[List[int]],
               model_name_or_checkpoin_path: str,
               batch_size: Optional[int] = 8,
               eval_metrics: Optional[List[str]] = ["f1", "precision", "recall", "accuracy"],
               tokenizer_name: Optional[str] = None,
               device_type: Optional[str] = None):
    import evaluate
    from collections import Counter
    from more_itertools import batched
    
    # load inference model
    model = InferenceModel(model_name_or_checkpoin_path=model_name_or_checkpoin_path,
                            tokenizer_name=tokenizer_name,
                            device_type=device_type)
    predicted_classes_ids: List[List[str]] = []
    for batch_text in tqdm(batched(tokens, batch_size), desc="Testing...", total=len(tokens)//batch_size):
        predicted_classes_ids.extend(model.inference(texts=batch_text))
    # convert labels_tokens to labels
    labels_tokens = [[model.id2label[label] for label in labels] for labels in labels_tokens]
    # check alignment between labels_tokens and predicted_classes_ids, remove them if they are not aligned
    true_labels_tokens, predicted_labels_tokens = [], []
    for l_tokens, pred_classes_id in zip(labels_tokens, predicted_classes_ids):
        if len(l_tokens) == len(pred_classes_id):
            true_labels_tokens.append(l_tokens)
            predicted_labels_tokens.append(pred_classes_id)
    # compute metrics
    seqeval_score = evaluate.load("seqeval")
    eval_metrics = {**(seqeval_score.compute(predictions=predicted_labels_tokens, references=true_labels_tokens))}
    return eval_metrics 
    