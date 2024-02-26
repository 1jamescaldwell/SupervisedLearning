This document outlines a supervised learning process involving the generation of synthetic data, fitting polynomial regression models, and evaluating their performance using mean squared error (MSE). The script explores the effects of varying parameters such as sample size and standard deviation on model fitting and prediction accuracy.

## Script Contents
The script consists of the following sections:
  
  1. **Data Generation**: Functions to simulate x and y values from normal distributions and define the true mean function.
2. **Data Visualization**: Scatterplots of the training data along with the true regression line.
3. **Model Fitting**: Polynomial regression models (linear, quadratic, cubic) fitted to the training data.
4. **Model Evaluation**: Prediction of test data using the trained models and calculation of MSE for each model.
5. **Simulation and Analysis**: Simulation of multiple evaluations with varying parameters to explore model performance under different conditions.

## Observations
- Increasing the standard deviation (sigma) introduces more noise, making the data less representative of the true model form.
- Larger sample sizes (n) provide a clearer picture of the data trends, leading to more accurate model fitting.
- The best model for prediction may not always be the true model form, especially when the data is highly variable.

For further details and code implementation, refer to the R script.
