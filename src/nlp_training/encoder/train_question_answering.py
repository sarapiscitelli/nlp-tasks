import numpy as np
import evaluate

from typing import Any, Dict, List, Optional
from functools import partial
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DefaultDataCollator,
    AutoModelForQuestionAnswering,
    Trainer,
    EvalPrediction
)

from ..callbacks import PrinterCallback

metric_eval = evaluate.load("squad_v2")

def compute_metrics(tokenizer, p: EvalPrediction):
    predictions, labels, inputs = p
    # predictions[0] is the start answer logits, predictions[1] is the end answer logits
    # for predictions get the index with the highest value as the answer index
    # select the input tokens from the start answer index to the end answer index
    predict_answer_tokens = [inputs[idx][answer_start_index : answer_end_index + 1]
        for idx, (answer_start_index, answer_end_index) in 
        enumerate(np.dstack((np.argmax(predictions[0], axis=-1), np.argmax(predictions[1], axis=-1)))[0,:])]
    true_answer_tokens = [inputs[idx][answer_start_index : answer_end_index + 1]
        for idx, (answer_start_index, answer_end_index) in
        enumerate(np.dstack((labels[0], labels[1]))[0,:])]
    # decode the tokens to get the answer text
    answers_labels: List[str] = tokenizer.batch_decode(true_answer_tokens)
    answers_predictions: List[str] = tokenizer.batch_decode(predict_answer_tokens)
    # compute the metrics
    predictions = [{'prediction_text': el, 'id': str(idx), 'no_answer_probability': 0.0} for idx, el in enumerate(answers_predictions)]
    references = [{'id': str(idx), 'answers': {'text': [el], 'answer_start': [labels[0][idx]]}} for idx, el in enumerate(answers_labels)]
    score = metric_eval.compute(predictions=predictions, references=references)
    return {
        "exact_match": score["exact"],
        "f1": score["f1"],
    }


def check_dataset(dataset: Dataset):
    """
    Dataset({
        features: ['context', 'question', 'answers'],
        num_rows: 5000
    })

    Args:
        dataset (Dataset): _description_
    """
    if 'context' not in dataset.features:
        raise ValueError("Dataset should have a feature called 'context' for the input context")
    else:
        # check if the context is a List[str]
        if not isinstance(dataset['context'], list):
            raise ValueError("The context should be a List[str]")
    if 'question' not in dataset.features:
        raise ValueError("Dataset should have a feature called 'question' for the input question")
    else:
        # check if the question is a List[str]
        if not isinstance(dataset['question'], list):
            raise ValueError("The question should be a List[str]")
    if 'answers' not in dataset.features:
        raise ValueError("Dataset should have a feature called 'answers' for the input answers")
    else:
        # check if the answers is a List[Dict[str, List[int]]]
        # e.g. dataset['answers'][0] = {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
        if not isinstance(dataset['answers'], list):
            raise ValueError("The answers should be a List[Dict[str, List[int]]]")
        else:
            if not isinstance(dataset['answers'][0], dict):
                raise ValueError("The answers should be a List[Dict[str, List[int]]]")
            else:
                if 'text' not in dataset['answers'][0]:
                    raise ValueError("The answers should be a List[Dict[str, List[int]]]")
                else:
                    if not isinstance(dataset['answers'][0]['text'], list):
                        raise ValueError("The answers should be a List[Dict[str, List[int]]]")
                    else:
                        if 'answer_start' not in dataset['answers'][0]:
                            raise ValueError("The answers should be a List[Dict[str, List[int]]]")
                        else:
                            if not isinstance(dataset['answers'][0]['answer_start'], list):
                                raise ValueError("The answers should be a List[Dict[str, List[int]]]")
    return True


def train_encoder(
    base_model_name_or_path: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_tokens_length: int,
    training_args: TrainingArguments,
    enable_printer: Optional[bool] = True,
):

    def _tokenize_data(examples: Dict[str, List[Any]]):
        """Tokenize data
        Args:
            examples (Dict[str, List[Any]]): examples to tokenize
                e.g. {'text': ['example 1', 'example 2', ...], 
                      'label': [0, 1, ...]}
        """
        questions = [q.strip() for q in examples[dataset_question_field]]
        inputs = tokenizer(
            questions,
            examples[dataset_context_field],
            max_length=max_tokens_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples[dataset_answer_field]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    dataset_question_field = "question"
    dataset_context_field = "context"
    dataset_answer_field = "answers"

    check_dataset(train_dataset)
    check_dataset(eval_dataset)

    # set up callbacks
    callbacks = []
    if enable_printer:
        callbacks.append(PrinterCallback(train_dataset=train_dataset,
                                         eval_dataset=eval_dataset,
                                         max_tokens_length=max_tokens_length,
                                         ))
        
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    # tokenize data
    train_dataset = train_dataset.map(_tokenize_data, 
                                      batched=True,
                                      remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(_tokenize_data, 
                                    batched=True,
                                    remove_columns=eval_dataset.column_names)

    # load data collator
    data_collator = DefaultDataCollator(return_tensors="pt")
    
    model =  AutoModelForQuestionAnswering.from_pretrained(
        base_model_name_or_path
    )
    training_args.include_inputs_for_metrics = True # needed for compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=partial(compute_metrics, tokenizer),
    )

    trainer.train()
