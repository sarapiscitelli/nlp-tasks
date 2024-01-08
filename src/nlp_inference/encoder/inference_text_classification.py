import torch

from typing import List, Optional
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class InferenceModel:

    def __init__(self, model_name_or_checkpoin_path: str,
                 tokenizer_name: Optional[str] = None,
                 device_type: Optional[str] = None) -> None:
        if tokenizer_name is None:
            tokenizer_name = model_name_or_checkpoin_path
        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name_or_checkpoin_path,
                                                                         device_map=device_type)
        if 'id2label' not in self.model.config.__dict__:
            raise ValueError(f"The model must be a sequence classification model, 'id2label' is not found in the model config = {self.model.config.__dict__}")
        self._id2label = self.model.config.id2label
        self._model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_checkpoin_path)
    
    @property
    def id2label(self):
        return self._id2label

    @property
    def model(self):
        return self._model

    def inference(self, texts: List[str]) -> List[int]:
        inputs = self.tokenizer(texts,
                                padding="longest",
                                return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits

        # logits.shape = (batch_size, num_labels)
        predicted_class_id: List[int] = logits.argmax(dim=-1).tolist()
        return predicted_class_id


def evaluation(texts: List[str],
               labels_texts: List[int],
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
    predicted_classes_ids: List[int] = []
    for batch_text in tqdm(batched(texts, batch_size), desc="Testing...", total=len(texts)//batch_size):
        predicted_classes_ids.extend(model.inference(texts=batch_text))
    # compute metrics
    f1_score, precision_score, recall_score, accuracy_score = \
        evaluate.load("f1"), evaluate.load("precision"), evaluate.load("recall"), evaluate.load("accuracy")
    eval_metrics = {**(f1_score.compute(predictions=predicted_classes_ids, references=labels_texts, average="micro")), 
                    **(precision_score.compute(predictions=predicted_classes_ids, references=labels_texts, average="micro")),
                    **(recall_score.compute(predictions=predicted_classes_ids, references=labels_texts, average="micro")),
                    **(accuracy_score.compute(predictions=predicted_classes_ids, references=labels_texts))}
    # compute supportss
    eval_metrics["support"] = {model.id2label[int(k)]: v for k,v in Counter(labels_texts).items()}
    return eval_metrics 
