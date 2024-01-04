from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from datetime import datetime

from .utils import get_str_from_dict, get_trainable_parameters

class PrinterCallback(TrainerCallback):

    def __init__(self, train_dataset, eval_dataset, **kwargs) -> None:
        """Every kwargs elements will be logged"""
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.others_args = kwargs
        self.start_training_time = None
        # filenames for the output files
        self.log_history_filename = "log_history.txt"
        self.datasets_info_filename = "dataset_info.txt"
        self.model_info_filename = "model_info.txt"
        self.training_info_filename = "training_info.txt"

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        # write datasets_info
        output_file = Path(args.output_dir) / "datasets_info.txt"
        with open(output_file, "w") as f:
            f.write(f"Train dataset:\n{self.train_dataset}\n\n")
            f.write(f"Eval dataset:\n{self.eval_dataset}\n\n")
            # print some examples
            f.write(f"Train dataset examples:\n")
            for i in range(5):
                f.write(f"{get_str_from_dict(self.train_dataset[i])}\n")
            f.write(f"\nEval dataset examples:\n")
            for i in range(5):
                f.write(f"{get_str_from_dict(self.eval_dataset[i])}\n")
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        # write model_info
        output_file = Path(args.output_dir) / self.model_info_filename
        with open(output_file, "w") as f:
            f.write(f"Model:\n{kwargs['model']}\nTrainable params:{get_trainable_parameters(kwargs['model'])}\n\n")
            f.write(f"Model device: {kwargs['model'].device}\n\n")
            f.write(f"Model dtype: {kwargs['model'].dtype}\n")
        # write training_info
        output_file = Path(args.output_dir) / self.training_info_filename
        with open(output_file, "w") as f:
            #f.write(f"Tokenizer:\n{get_str_from_dict(kwargs['tokenizer'])}\n\n")
            f.write(f"Data collator:\n{get_str_from_dict(kwargs['train_dataloader'].collate_fn)}\n\n")
            f.write(f"Dataset Train:\n{kwargs['train_dataloader'].dataset}\n\n")
            for k, v in self.others_args.items():
                f.write(f"{k}: {get_str_from_dict(v)}\n\n")
            f.write(f"Training arguments:\n{get_str_from_dict(args.to_dict())}")

        # save start training time
        self.start_training_time = datetime.now()
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        end_training_time = datetime.now() - self.start_training_time
        # write training_info
        output_file = Path(args.output_dir) / self.training_info_filename
        with open(output_file, "a") as f:
            f.write(f"\n\nTraining time: {end_training_time}\n")
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        # write log_history
        output_file = Path(args.output_dir) / self.log_history_filename 
        with open(output_file, "w") as f:
            for el in state.log_history:
                f.write(f"{el}\n")
        pass
