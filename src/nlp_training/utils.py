import torch
import logging
import torch.nn as nn
import logging
import pprint

from typing import Tuple
from enum import Enum
from peft import PeftConfig, TaskType, LoraConfig, PromptTuningConfig, PromptEncoderConfig, IA3Config, PromptTuningInit


logger = logging.getLogger(__name__)


class LoadDtype(Enum):
    auto = "auto"
    float32 = torch.float32
    float16 = torch.float16
    int8 = torch.int8
    

def get_str_from_dict(input_dict, indent=3):
    pretty_printer = pprint.PrettyPrinter(indent=indent, depth=7)
    pretty_string = pretty_printer.pformat(input_dict)
    return pretty_string


def find_all_linear_names(model):
    import bitsandbytes as bnb

    if getattr(model, "is_loaded_in_8bit", False):
        cls = bnb.nn.Linear8bitLt
    elif getattr(model, "is_loaded_in_4bit", False):
        cls = bnb.nn.Linear4bit
    else:
        cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # lm_head is often excluded.
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def build_peft_config(peft_config: PeftConfig, model: torch.nn.Module):
    """This function adds to the model the PEFT components (prompt encoder, prompt tuning, etc.).

    Args:
        model (torch.nn.Module): the model to which add the PEFT components
        task_type (TaskType): the task type of the model (SEQ_CLS, SEQ_2_SEQ_LM, CAUSAL_LM, TOKEN_CLS, QUESTION_ANS, FEATURE_EXTRACTION)
        peft_config (LoraConfig | PromptTuningConfig | PromptEncoderConfig | IA3Config): the configuration of the PEFT components to add.

    Returns:
        torch.nn.Module: the model with the PEFT components added
    """

    peft_config.task_type = TaskType.CAUSAL_LM
    if isinstance(peft_config, LoraConfig):
        if peft_config.target_modules is None:
            peft_config.target_modules = find_all_linear_names(model)
            logger.info(
                f"Changing target modules to {peft_config.target_modules}")
        logger.debug(
            f"Adding a LoraConfig model with configuration = \n{get_str_from_dict(peft_config)}")
    elif isinstance(peft_config, PromptTuningConfig):
        peft_config.prompt_tuning_init = PromptTuningInit.TEXT
        peft_config.tokenizer_name_or_path = model.config._name_or_path
        logger.debug(
            f"Adding a PromptTuningConfig model with configuration = \n{get_str_from_dict(peft_config)}")
    elif isinstance(peft_config, PromptEncoderConfig):
        logger.debug(
            f"Adding a PromptEncoderConfig model with configuration = \n{get_str_from_dict(peft_config)}")
    elif isinstance(peft_config, IA3Config):
        logger.debug(
            f"Adding a IA3Config model with configuration = \n{get_str_from_dict(peft_config)}")
    else:
        raise ValueError(
            f"peft_config must be an instance of LoraConfig, PromptTuningConfig, PromptEncoderConfig or IA3Config, but got {type(peft_config)}")
    return peft_config


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns the number of trainable parameters and the total number of parameters in the model.
    """
    trainable_count = 0
    total_count = 0

    for _, param in model.named_parameters():
        param_count = param.numel()

        # If using DS Zero 3 and the weights are initialized empty
        if param_count == 0 and hasattr(param, "ds_numel"):
            param_count = param.ds_numel

        # Adjust for the design of 4-bit linear layers from bitsandbytes
        # see https://github.com/huggingface/peft/issues/1023#issuecomment-1765348615
        if param.__class__.__name__ == "Params4bit":
            param_count *= 2

        total_count += param_count
        if param.requires_grad:
            trainable_count += param_count

    return trainable_count, total_count


def get_trainable_parameters(model: nn.Module) -> str:
    """
    Prints the number of trainable parameters in the model. 
    """
    trainable_params, all_param = count_parameters(model=model)

    return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
