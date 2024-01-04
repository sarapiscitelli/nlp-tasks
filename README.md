# nlp-tasks

The repository contains a series of natural language processing (NLP) tasks addressed using various methods and models.The intention is to have a solid codebase for experimenting with different models on various standard datasets, therefore the repo has more of an experimental purpose.

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
