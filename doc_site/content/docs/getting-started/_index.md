---
title: Getting Started
type: docs
weight: 100
prev: docs/
next: docs/installation
---


PETsARD's target audience is any data user, meaning we do not expect users to have engineering knowledge, particularly Python programming skills. Users only need to use the following three lines of Python code and can focus on writing configuration files:

```python
from petsard import Executor

exec = Executor(config=yaml_path)
exec.run()
```

Our only public class is `Executor`, and users interact with PETsARD through YAML configuration files. The required configuration files include:

1. **PETsARD YAML**: One synthesis execution corresponds to one configuration file
2. **Schema YAML**: One data table corresponds to one Schema configuration
3. **Constraints YAML** (optional): One data table corresponds to one constraint configuration

## What is YAML?

YAML (YAML Ain't Markup Language) is a human-readable data serialization format that PETsARD uses for experiment configuration.

- **Easy to read and write**: Uses indentation and concise syntax, understandable without programming background
- **Clear structure**: Expresses hierarchical relationships through indentation, visually clear at a glance
- **Supports multiple data types**: Strings, numbers, booleans, lists, objects, etc.

The following scenarios guide you in choosing the right YAML configuration:

1. **[Basic Usage: Data Synthesis with Default Parameters](default-synthesis)**

  - When you only need basic data synthesis
  - For simple privacy-enhanced synthetic data generation

2. **[Basic Usage with Evaluation: Data Synthesis and Evaluation with Default Parameters](default-synthesis-default-evaluation)**

  - When you need both synthesis and comprehensive evaluation
  - Includes protection, fidelity, and utility assessments

3. **[Evaluation of External Solutions: Evaluation of External Synthetic Data with Default Parameters](external-synthesis-default-evaluation)**

  - When you have pre-synthesized data
  - For evaluating existing privacy-enhanced solutions

The following scenarios use the extended features of this edition. Prepare the environment with [Extended Features Setup](../installation/extended-features) first:

4. **[Semi-structured Data: JSON Input and Output with AdvGAN Synthesis](json-advgan-synthesis)**

  - When your data is stored as JSON records
  - Synthesizes data with the AdvGAN synthesizer and outputs the synthetic data and reports as JSON

5. **[Unstructured Data: Local LLM Text Synthesis and Text Evaluation](llm-text-synthesis)**

  - When your data contains free-text (natural language) columns
  - Generates synthetic text with a locally hosted large language model and evaluates it with the text evaluator

6. **[Missing Data: Missing-Data Compensation with Median, KNN and MICE](missing-compensation)**

  - When numeric feature columns contain missing values
  - Fits a compensation strategy on public or external data and fills the missing cells of the local data

Simply choose the scenario that matches your needs, prepare the corresponding YAML configuration, and run the code above.