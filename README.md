# VagueBench

> *"What can be said at all can be said clearly; and whereof one cannot speak thereof one must be silent."*
Wittgenstein

### Overview

This repository contains the evaluation code for VagueBench, a project funded by the **Cosmos Institute**. In philophical literature, some sentences are classified as vague. This entails that one is not to assign determinate truth-values to these statements. When one does, one commits what I call a vagueness error. 

VagueBench measures the extent to which LLMs can serve as helpful truth-seeking assistants when it comes to making humans aware of their vagueness errors. An article explaining VagueBench in more depth may be published soon.

### Methodology

The benchmark utilizes a disagreement scenario between two humans regarding a vague sentence.

1. **Person A** asserts a vague proposition (e.g., *"7am is early"*).
2. **Person B** asserts that what Person A says is false.
3. The **LLM** is asked to adjudicate.

A "failure" is recorded if the model sides with one without qualification. A "success" is recorded if the model acknowledges that one is not to assign a strict truth-value to the sentence in question.

### Benchmark Results

<img width="3569" height="2369" alt="vaguebench_results" src="https://github.com/user-attachments/assets/0ba60493-27fe-4b84-b700-9d104908502b" />
*Figure 1: scores of LLMs on VagueBench, representing the proportion of prompts to which the LLM responded with an acknowledgement that no truth-value should strictly be assigned. The scores are averaged over three runs. *

### Dataset Access

To prevent data contamination, the dataset is not hosted in this repository but on HuggingFace. One can find it at wvinke/VagueBench (with gated access).

The evaluation code automatically downloads the dataset from HF when HF_TOKEN is set in a .env file. The code is made to be compatible with ARM64 architecture. 

**Note:**

* There is only a 'test' split.
* Please refrain from making the dataset public without gated access.

### Usage

This repository provides code to run the benchmark against models through OpenRouter's API. 

#### 1. Installation

```bash
git clone https://github.com/WesselVinke/VagueBench.git
cd VagueBench
pip install -r requirements.txt
```

#### 2. Accessing the Data

Gain access to the data on HF here: 

https://huggingface.co/datasets/wvinke/VagueBench

Create a .env file in the local repository and add your HF Access Token as HF_TOKEN. This will allow the code to download the dataset through your HF account.

#### 3. Running the Evaluation

Add your OpenRouter API key in the .env file as OPENROUTER_API_KEY.

At the top of proposition_evaluator.py, one can configure the number of parallel agents, the number of repetitions per vague proposition, retry logic, and the models for completion and grading. 

Next one can run the proposition_evaluator.py file.
