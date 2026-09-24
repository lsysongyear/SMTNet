# libribrain-speech detection

## Project Environment

This project is designed for speech detection using the LibriBrain dataset. It relies on a Python-based environment managed via Conda. To set up the environment, follow these steps:

- **Dependencies**: The project requires libraries such as PyTorch, NumPy, Scikit-learn, and Matplotlib for model training, evaluation, and visualization.
- **Setup**: Use the provided `environment.yml` file to create the environment by running the following command in your terminal:
  ```bash
  conda env create -f environment.yml


## Directory and File Descriptions

- **`configs/`**  
  Path 'config/speech/my_run' contains configuration files for the project. 

- **`libribrain/`**  
  Original LibriBrain dataset. Please name the original dataset located in the task1 directory as "libribrain".

- **`speech_code/`**  
  Includes the source code for the speech detection implementation.
  - **`models/`**: Contains the models we used in the project.
  - **`data_process.py`**: Used to generate the data that has been downsampled to 100Hz and selected grad magnetometer channels. The data is saved in the corresponding directory of libribrain.
  - **`speech_utils.py`**: Defines a new holdout data class, logic for training and validation and a function to generate the holdout CSV file.
  - **`utils.py`**: Defines the data classes used for model training.
  - **`train.py`**: The main logic for model training.

- **`results/`**  
  Save the results for each training session.

- **`ch_types.npz`**  
  Stored the category information of MEG channels.

- **`environment.yml`**  
  A YAML file defining the project environment, including dependencies and package versions required to run the code (e.g., using Conda).

- **`Readme.md`**  
  This file, providing an overview of the project, installation instructions, and usage guidelines.

## Usage

To get started with the project:
1. Change to the `task1` directory by running the following command:
     ```bash
     cd /path/to/your/project/task1
2. Generate the data file 
  - First, you need to place the original LibriBrain dataset in the task1 directory and name it "libribrain":
    - libribrain
      - COMPETITION_HOLDOUT
      - Sherlock1/derivatives
        - events
        - serialised
      - Sherlock2
      - Sherlock3
      - Sherlock4
      - Sherlock5
      - Sherlock6
      - Sherlock7
  - Then, run:
    ```bash
     python speech_code/data_process.py
  - Alternatively, you can specify the path as an existing data path and modify the data path in `configs/speech/my_run/config.yaml`.
  
3. run:
     ```bash
     python speech_code/train.py --config=configs/speech/my_run/config.yaml --search-space=configs/speech/my_run/search-space.yaml --holdout=True --run-index=0 

Converged in approximately 10 epochs.

## Supplementary information

1. Due to the outdated CUDA version on the server we are using, there may be some compatibility issues with the environment. If you encounter any environment-related problems, please contact us immediately.
2. We provide a convenient tuning interface that allows you to try different hyperparameter combinations.