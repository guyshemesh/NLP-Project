# Layer Duplication in Large Language Models (LLMs)

This repository contains code and data for the project **"Exploring Layer Duplication in Large Language Models"**, which explores the effects of duplicating frozen transformer layers in pre-trained LLMs, such as Pythia and LLaMA3. The goal of this research is to investigate whether duplicating multi-head self-attention (MHSA) layers in these models can enhance performance across various complex NLP tasks without additional training.

## Project Overview

In this project, we:
- Developed custom versions of the `pythia-70m-deduped` model by duplicating selected MHSA layers.
- Evaluated these models across a wide range of `BIG-bench` tasks to assess performance changes due to layer duplication.
- Conducted limited evaluations on larger models (`pythia-6.9b` and `llama-3-8B`) to gain insights into scalability.

The findings indicate that duplicating specific layers in frozen LLMs can yield task-specific performance improvements, suggesting a potential for cost-effective model enhancement.

## Repository Structure

├── models/                      # Scripts to create custom models by duplicating layers
├── tasks/                       # Code for loading tasks, running evaluations, and saving results
├── results/                     # Output files with performance metrics for each task
└── README.md                    # Project documentation (this file)

### Key Directories

- **models/**: Contains scripts to load pre-trained models and duplicate selected MHSA layers. 
- **tasks/**: Includes scripts to load tasks from `BIG-bench`, evaluate the models on these tasks, and save the results to CSV files.
- **results/**: Stores the output files with results for each model/task combination.

## Group Members

- Neo Eyal, neoedan@gmail.com
- Adi Shani, adishani1@mail.tau.ac.il
- Milana Yakubov, milanay1@mail.tau.ac.il
- Guy Shemesh, Guyshemesh@mail.tau.ac.il
Mentor: Dr. Mor Geva, morgeva@tauex.tau.ac.il
