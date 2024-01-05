# nlp-tasks

The repository contains a series of natural language processing (NLP) tasks tackled using Transformers models and standard open-source datasets.    
Its implementation primarily utilizes PyTorch, heavily relying on the [HuggingFace Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) libraries.   
The intention is to have a codebase for experimenting with different models on various standard datasets, therefore the repository has more of a demonstrative or experimental purpose.  
Tasks included are:  

- [✅] **Text Classification**
- [✅] **Named Entity Recognition**
- [✅] **Question Answering**
- [✅] **Summarization**
- [✅] **Causal Llm**
- [] **Instruction Fine tuning Llm**

## How to Run

1. Initial setup  
   1.1 Create a new conda environment to install the dependencies, and activate it:
   ```bash
   conda create -n nlp-tasks-env python=3.11 -y
   conda activate nlp-tasks-env
   ```
    1.2 Install the dependencies:
    ```bash
    pip install git+https://github.com/sarapiscitelli/nlp-tasks/
    ```
    1.3 Clone the repository to get the scripts:   
    ```bash
    git clone https://github.com/sarapiscitelli/nlp-tasks.git
    ```

2. Run the experiments   
    Training:
    ```bash
    python scripts/train/<task name>.py
    ```
    Evaluation:
    ```bash
    python scripts/evaluate/<task name>.py --model_name_or_checkpoin_path <path_to_model>
    ```
