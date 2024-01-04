import torch

from typing import List, Optional
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class InferenceModel:

    def __init__(self, model_name_or_checkpoin_path: str,
                 tokenizer_name: Optional[str] = None,
                 device_type: Optional[str] = None) -> List[str]:
        if tokenizer_name is None:
            tokenizer_name = model_name_or_checkpoin_path
        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_checkpoin_path, device_map=device_type)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_checkpoin_path)

    def inference(self, questions: List[str], contexts: List[str]) -> List[str]:
        inputs = self.tokenizer(questions, contexts,
                                padding="longest",
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs)
        # logits.start_logits.shape == (batch_size, input_length) = inputs['input_ids'].shape
        # logits.end_logits.shape == (batch_size, input_length) = inputs['input_ids'].shape
        answer_start_index: List[int] = logits.start_logits.argmax(dim=-1).tolist()
        answer_end_index: List[int] = logits.end_logits.argmax(dim=-1).tolist()
        answer_tokens: List[str] = [self.tokenizer.decode(inputs.input_ids[i, answer_start_index[i] : answer_end_index[i] + 1])
                                    for i in range(len(questions))]
        return answer_tokens

    
def evaluation(questions: List[str],
               contexts: List[str],
               answers_labels_texts: List[str],
               answer_starts: List[int],
               model_name_or_checkpoin_path: str,
               batch_size: Optional[int] = 8,
               eval_metrics: Optional[List[str]] = ["f1", "exact_match"],
               tokenizer_name: Optional[str] = None,
               device_type: Optional[str] = None):
    import evaluate
    from more_itertools import batched
    
    # load inference model
    model = InferenceModel(model_name_or_checkpoin_path=model_name_or_checkpoin_path,
                            tokenizer_name=tokenizer_name,
                            device_type=device_type)

    predicted_answers: List[str] = []
    for batch in tqdm(batched(zip(questions, contexts), batch_size), desc="Testing...", total=len(questions)//batch_size):
        questions_batch, contexts_batch = zip(*batch)
        predicted_answers.extend(model.inference(questions=questions_batch,
                                                     contexts=contexts_batch))

    predictions = [{'prediction_text': el, 'id': str(idx), 'no_answer_probability': 0.0} for idx, el in enumerate(predicted_answers)]
    references = [{'id': str(idx), 'answers': {'text': [el], 'answer_start': [answer_starts[idx]]}} for idx, el in enumerate(answers_labels_texts)]
    
    # compute metrics
    metric_eval = evaluate.load("squad_v2")
    eval_metrics = {**(metric_eval.compute(predictions=predictions, references=references))}
    return eval_metrics 