from datasets import Dataset
from typing import Dict, List, Any
from functools import partial


def create_instruction_for_generation(dataset: Dataset,
                                      instruction_template: str,
                                      placeholder_variables: Dict[str, str],
                                      input_feat: str) -> Dataset:
    """Prepend an instruction to the text in the dataset.
    All the text in the input_feat column will be prepended with the instruction_template and 
    response_template and the placeholder variables are replaced with the corresponding values in the dataset.

    Args:
        dataset (Dataset): the dataset to modify
        instruction_template (str): the template of the instruction to prepend in front of the texts
            NOTE: The template should contain the placeholder variables in the form of {{$variable_name}} otherwise the placeholder will not be replaced.
        placeholder_variables (Dict[str, str]): the variables to replace in the instruction and response template
        end_instruction_template (str): the token that means the end of the instruction
        input_feat (str): the name of the input feature to modify

    Returns:
        Dataset: a new dataset with the modified input feature
    """

    def _create_dataset(examples, instruction_template, placeholder_variables, input_feat) -> List[Dict[str, Any]]:
        instructions: list = [instruction_template] * \
            len(examples[list(examples.keys())[0]])
        for k, v in placeholder_variables.items():
            txts = examples[v]
            for idx, t in enumerate(txts):
                if "{{$"+k+"}}" in instruction_template:
                    instructions[idx] = instructions[idx].replace(
                        "{{$"+k+"}}", str(t))

        return {input_feat: instructions}


    _create_dataset = partial(_create_dataset, instruction_template=instruction_template,
                              placeholder_variables=placeholder_variables, input_feat=input_feat)
    dataset = dataset.map(_create_dataset, batched=True,
                          desc="Creating instruction for generation")

    return dataset
