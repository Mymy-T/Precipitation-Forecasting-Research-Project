This is the code used in the research project titled *Investigating the Sensitivity of Precipitation Forecasting Models to Predictor Location Number and Concentration*. Specifically, this project explored how the precipitation classification performance of logistic regression and LSTM models changed with different input spatial configurations. The results show that the models responded differently to different input configurations, and that including more predictor locations generally improved performance. 

![Research Poster](https://i.postimg.cc/MT9cmyQC/Poster-Anon.png)
<p align=center>Research Poster</p>
 
## Project Layout

The folders hold the datasets, final models, and run logs (loss curves). A brief description of each .py file is given in the table:

| File | Function |
|--|--|
| `analysis.py` | Run the statistical tests on the experimental results and record them as a .md file |
| `data_converter.py` | Loads and processes the raw .grib data files before saving them as .csv files |
| `data_preprocessor.py` | Splits the data into train, validation, and test datasets. Also performs preprocessing, such as feature standardization |
| `dataset.py` | A customized PyTorch dataset class that loads examples and creates batches |
| `dataset_creator.py` | Contains a function that builds the projected 2D trajectories from the 3D trajectories; also interpolates both to a desired time step |
| `dataset_composition.py` | Determines the number of wet and dry hours in the training dataset; this is used to determine loss imbalance weights |
| `hyperparameter_tester.py` | Automatically performs the sequential hyperparameter tuning and logs the results into a file for records  |
| `main_experiment.py` | Automatically performs the model training & evaluations across all spatial input configurations; logs all results to a file for later analysis |
| `model.py` | Defines the structure of the LSTM model |
| `model_weight_norm_checker.py` | Used to double check if the weights different between two similar weight decay settings since both resulted in the same performance |
| `performance_log_manager.py` | Contains a class and specific functions for training and evaluating the LSTM and logistic regression models; its `run_and_log` function is called once per trial, and this runs the trial and saves the results  |
| `random_guessing.py` | Runs random guessing on the test dataset to see how a model without information performs on the task |
| `trainer.py` | Contains the train and evaluation loops for the LSTM; also contains the function that actually executes the random guessing loop called by `random_guessing.py` |
| `variables.py` | Contains data classes specifying the configuration of the project, as well as the paths to data and results files |

## Technologies
This project is written in Python. The specific libraries utilized include PyTorch, Pandas, XArray, and Statsmodels.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMTQ0ODEyNDFdfQ==
-->