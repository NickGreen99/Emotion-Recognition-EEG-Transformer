# EEG-Based Emotion Recognition with Transformers

This repository contains the Python code implementation for my dissertation thesis in the Department of Electrical and Computer Engineering at the University of Patras. The thesis is based on the paper *"Transformers for EEG-Based Emotion Recognition: A Hierarchical Spatial Information Learning Model"* by Zhe Wang et al.

## Repository Structure

- `deap_preprocessing.py`: Handles the DEAP dataset,the preprocessing of EEG data and feature extraction.
- `deap_transformer.py`: Contains code for training the model, evaluating performance, and visualizing results. 
- `deap_transformer_classes.py`: Defines the transformer-based model used for emotion recognition, following the architecture described in the paper.
- `Diploma_Thesis.docx`: My dissertation thesis detailing the research, methodology, experiments, and results.
- `requirements.txt`: Text file containing all the required libraries.


## Prerequisites

- DEAP dataset: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
- Python 3.x
- Install required dependencies using the following command:

  ```bash
  pip install -r requirements.txt
