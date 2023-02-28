# Hybrid-Battery-Feature-Selection

This repository contains an analysis of different machine learning models using feature selection for Hybrid Battery State Estimation models. The models consist of physics-based models (PBM) and machine learning models using Pytorch. The code is implemented in Python with the use of libraries such as Pandas and Pytorch.

## Getting Started

### Prerequisites
- Python 3.x
- Pytorch
- Pandas
- Pickle
- Utils library from the repository
- Data folder containing the Clean_Data_Full_SOC.csv file.

### Installing
1. Clone or download the repository to your local machine.
2. Navigate to the root folder of the repository.
3. Run the main script with the desired model name (lstm, gru, rnn, or mlp), k value for feature selection, early stopping option and store option to store the model and feature selection filter.

`python main.py --model_name lstm --k 20 --early_stopping True --store True`


## Usage
The script allows you to train and test different machine learning models on the provided dataset. The feature selection is used to select the most relevant features for the battery state estimation. The models can be trained with or without early stopping. The performance of the models can be evaluated by the Root Mean Squared Error (RMSE). The script also allows you to store the best model and the feature selection filter after training. The training history can be visualized with the use of a plot.

## Data
The dataset is provided in the Data folder and it is a .csv file named Clean_Data_Full_SOC.csv which should include all the data used for the analysis and tests.

## Contributing
We welcome contributions to this repository. If you are interested in contributing, please fork the repository and submit a pull request with your proposed changes. Be sure to also update the documentation in the README as necessary.

## Credits
This project was developed by [Your Name or team]. The dataset was provided by [Source of the data].

## References
- [Reference 1]
- [Reference 2]

