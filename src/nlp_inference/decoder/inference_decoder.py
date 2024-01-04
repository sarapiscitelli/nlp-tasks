import gc
import torch

from enum import Enum
from typing import List, Optional
from tqdm import tqdm
from transformers import (
    GenerationConfig, 
    BitsAndBytesConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM
)

class LoadDtype(Enum):
    auto = "auto"
    float32 = torch.float32
    float16 = torch.float16
    int8 = torch.int8
    

class InferenceDecoder:
    """This class is a wrapper for the generation of text from a causal language model. """

    def __init__(self,
                 base_model_or_path: str,
                 tokenizer_name: Optional[str] = None,
                 adapter_model_path: Optional[str | None] = None,
                 device_type: str = "auto",
                 load_dtype: str = "auto",
                 quantization_config: Optional[BitsAndBytesConfig | None] = None) -> None:
        """Load a causal language model from HuggingFace or a local path, according to the configuration.
        """
        if tokenizer_name is None:
            tokenizer_name = base_model_or_path

        # load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True)
        #self._tokenizer.add_bos_token = False
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        
        # load model
        self._model = AutoModelForCausalLM.from_pretrained(base_model_or_path,
                                                            device_map=device_type,
                                                            quantization_config=quantization_config,
                                                            torch_dtype=LoadDtype[load_dtype].value)
        self._model = self._model.eval()

        if adapter_model_path is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model, adapter_model_path)
            self._model = self._model.eval()
        
    @property
    def model(self):
        return self._model

    def generate(self, prompt: str, generation_config: GenerationConfig | None) -> str:
        """Generate text from a causal language model.

        Args:
            prompt (str): the prompt to use for the generation
            generation_config (GenerationConfig): the generation configuration to use

        Returns:
            str: the generated text
        """
        input = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        if generation_config is None:
            generation_config = GenerationConfig()
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = None
        if generation_config.temperature is not None:
            generation_config.do_sample = True
        generation_config.eos_token_id = self._tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = self._model.generate(input_ids=input['input_ids'], 
                                                attention_mask=input['attention_mask'],
                                                pad_token_id=self._tokenizer.eos_token_id,
                                                generation_config=generation_config)
        output_ids = output_ids.detach().cpu()
        input['input_ids'] = input['input_ids'].detach().cpu()
        # remove all tokens that are part of the input_ids tokens
        output_ids = output_ids[:, input['input_ids'].shape[1]:]
        output_texts: List[str] = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        del output_ids
        del input
        gc.collect()
        torch.cuda.empty_cache()
        return output_texts[0]


def evaluation(texts: List[str],
               text_labels: List[str],
               model_name_or_checkpoint_path: str,
               batch_size: Optional[int] = 8,
               generation_config: Optional[GenerationConfig] = None,
               eval_metrics: Optional[List[str]] = ["rouge_score"],
               tokenizer_name: Optional[str] = None,
               device_type: str = "auto",
               load_dtype: str = "auto",
               quantization_config: Optional[BitsAndBytesConfig | None] = None):

    import evaluate
    from more_itertools import batched

    if generation_config is None:
        generation_config = GenerationConfig( max_new_tokens=100, temperature=0.1, do_sample=True)
    # load inference model
    model = InferenceDecoder(base_model_or_path=model_name_or_checkpoint_path,
                             tokenizer_name=tokenizer_name,
                             device_type=device_type,
                             load_dtype=load_dtype,
                             quantization_config=quantization_config)

    predicted_texts: List[str] = []
    for batch in tqdm(batched(texts, batch_size), desc="Testing...", total=len(texts)//batch_size):
        for t in batch:
            predicted_texts.append(model.generate(prompt=t, generation_config=generation_config))

    # compute metrics
    rouge_metric = evaluate.load("rouge")
    eval_metrics = {**(rouge_metric.compute(predictions=predicted_texts, references=text_labels))}
    return eval_metrics
