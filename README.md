**Explanation of the structure:**

- `src/`: Contains all the source code files.
- `data/`: Contains all the data used in the project.
- `docs/`: Contains documentation like reports and presentations.
- `results/`: Contains the outputs of your analyses.
- `tests/`: Contains testing scripts.
- `.gitignore`: Lists files/directories to ignore.
- `LICENSE`: Contains the license information.
- `README.md`: This file, containing information about the project.

## Project Report Format

### Project Title

Steel Production Data
### Abstract

The goal of this project is to predict the quality of steel production based on process sensor data. Accurate prediction of steel output can help reduce waste, improve process control, and ensure consistent product quality in industrial environments.

### Introduction

**Background**

Predictive modeling in industrial production has been widely studied in recent years, particularly in the context of Industry 4.0 and smart manufacturing. Existing work can generally be categorized according to (1) sensing technology and (2) machine learning methodology.

Sensing-based approaches leverage data from multiple sensor modalities (e.g., temperature, pressure, chemical composition) to detect anomalies or predict production outcomes. For example, acoustic emission and thermal sensors have been used for quality inspection, while radar and camera-based monitoring systems detect surface defects in real time.

Machine learning-based approaches employ algorithms such as Linear Regression, Support Vector Machines, and Random Forests to model nonlinear relationships in process data. More recent works have explored deep learning architectures such as convolutional neural networks, which can automatically learn complex feature representations from raw sensor streams.

However, these approaches often struggle with interpretability and require large amounts of labeled data. Some hybrid systems combine classical feature engineering (e.g., statistical features, PCA) with neural network-based regressors to balance accuracy and explainability.

In this project, we developed a fully connected (dense) neural network (MLP) for predicting the steel output variable between 0 and 1, using 21 normalized process parameters as input features. The network was trained, validated, and evaluated on normalized production data.

**Objectives**

The primary objectives of this project are:

1. Data Quality Assessment: Conduct comprehensive exploratory data analysis to identify missing values, duplicates, outliers, and feature correlations in the steel production dataset.

2. Baseline Model Establishment: Implement and evaluate a baseline regression model (linear) to establish performance benchmarks for comparison.

3. Implement various ML models to beat the baseline model used as a benchmark. Models should include classic classifiers such as RFClassifier, KNN, LogisticRegressor as well as classic regressors such as RFRegressor, SVM's and Gaussian Processes 

4. Neural Network Development: Design and train optimized deep neural network architectures incorporating modern regularization techniques (BatchNormalization, Dropout) and adaptive learning strategies (EarlyStopping, ReduceLROnPlateau).

5. Performance Target: Achieve a coefficient of determination (R²) of at least 0.5 on the test dataset, demonstrating meaningful predictive capability for steel production output.

6. Model Interpretability: Analyze feature importance and model behavior through correlation analysis, learning curves, and prediction visualizations to ensure practical applicability.

7. Robust Prediction: Implement Huber loss and other robust techniques to handle potential outliers and ensure stable predictions across various production conditions.

## Methods

**Data Acquisition**

For those that choose their own project, detail where and how you obtained your data, including a brief description of the data.

**Data Analysis**

The data analysis pipeline consisted of multiple stages:

**1. Exploratory Data Analysis (EDA)**

- Missing value analysis: identified and quantified null values across all columns, visualized null fraction distributions
![Missing Value Analysis](results/figures/MissingValues_Columns.png)

- Duplicate detection: identified and analyzed duplicate rows to assess data quality
![Duplicates in Rows](results/tables/n_Duplicates.png)
Summary: No Duplicates have been found before replacing NaN's with the arithmetic mean of columns

- Distribution analysis: created histogram of output and pairplots (using seaborn and matplotlib) to visualize feature and target distributions
![alt text](results/figures/output_hist_totalnumbers.png)
![alt text](results/figures/outputvsinput1to21_histogramm.png)

- Statistical summary: computed a heatmap for all features
![alt text](results/figures/heatmap_correlations.png)






**2. Data Preprocessing**
- Missing value imputation: No NaN Values have been found, regardless of n_fractions==0 NaN's are replaced with the mean in each column (for future use)

-Due to the nature of test and train_data (as seen in histogramm) the data of both sets have been concentuated to one dataset to properly model across the range [0-1]

- Train-test split: As stated above a randomized train_test_split has been implemented on the whole dataset

- Feature scaling: applied StandardScaler (fit on training data only) to normalize input features for neural network training -> This step is redundant as the given data is already normalized with a min-max transformation as stated in the .ipnyb notebook in template.
![alt text](results/figures/Min-Max_scaling.png)


**3. Machine Learning Models**

*Baseline Regression Model:*
- Linear Regression: established baseline performance using ordinary least squares
*Regression Models:*
- Random Forest Regressor: Evaluated based on R2-score using default hyperparameters

ADD THOSE TWO: -Gaussian Process: 
                -Mixed Gaussian Process

*Deep Learning Models:*
- Multi-layer perceptron (MLP) architectures with varying depths (128→64→1 and 256→128→64→32→1)
- Activation functions: ReLU  for hidden layers; linear, softplus for output layer
- Regularization: BatchNormalization and Dropout (rates 0.1-0.3) to prevent overfitting
- Loss functions: Mean Squared Error (MSE) and Huber loss for robust regression
- Optimizers: Adam with learning rate 1e-3, including adaptive scheduling via ReduceLROnPlateau
- Training callbacks: EarlyStopping (patience 12-15 epochs) to prevent overfitting and restore best weights

**4. Model Evaluation**
- Metrics: R² score, Root Mean Squared Error (RMSE)
- Learning curves: plotted training and validation loss over epochs to diagnose overfitting/underfitting
- Prediction visualization: scatter plots comparing ground truth vs. model predictions on test data

**Tools Used**

List and briefly describe the tools or technology stack used (e.g., programming languages, libraries, frameworks).

## Results

All models (due to their continous/discrete nature) were evaluated using the R2-Score and RMSE as standard measurments.
In the first few iterations it was worngly assumed that the given data is of categorical nature due to output column having 132 unique values on the training set. Thus the first few iterations of code utilising classifiers such as  Logistic Regression, KNN, RFClassifier which were written in an anaconda enviroment yielded very bad scores. These weren't implemented in github because of these aforementioned underwhelming results (R2-scores <0,1%).

After talks with Mr. Feith at CPS it was concluded that these data_entries were indeed of continous nature but due to scaling and/or accuarcy of the raw data appear discrete.

Linear Regression: 
![alt text](<results/figures/ground_truth vs predicted_lr.png>)
![alt text](results/figures/Model_Evaluation_lr.png)
Present the key findings of your project. Ensure to include visual representations of your results, like plots, graphs, or tables, and provide interpretations of these findings.

Random Forest Regressor:
![alt text](<results/figures/ground_truth vs predicted_rfreg.png>)
![alt text](results/figures/Model_Evaluation_rfreg.png)

DNN standard:
![alt text](<results/figures/ground_truth vs predicted_dnnstd.png>)
![alt text](<results/figures/loss vs epoch_dnnstd.png>)
![alt text](results/figures/Model_Evaluation_dnnstd.png)


DNN optimized:
![alt text](<results/figures/ground_truth vs predicted_dnnopt.png>)
![alt text](<results/figures/loss vs epoch_dnnopt.png>)
![alt text](results/figures/Model_Evaluation_dnnopt.png)

![Results of all Modles](results/figures/Scores.png)

**Findings**

[Briefly summarize the key findings of your analysis.]

**Visualizations**

[Insert images or use markdown to create tables. Make sure that visuals are clearly labeled and have appropriate captions.]

## Conclusion

Summarize the main points of your project, relating them back to your objectives. Discuss the implications of your findings and any limitations in your study. Provide any future directions for this work.

## License

For those that choose their own project, provide license details of your dataset e.g., this data is licensed under the [The License] - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments
- Mention any individuals or organizations that helped you in executing the project.
- Reference any research papers, data, or other resources that were crucial for the completion of your project. Do not forget to provide ChatGPT prompts you used for the project.

