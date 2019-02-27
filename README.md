# NLP-analysis-project

This project develops features and models to predict labels of all records in the file “testing_docs.txt” in the data folder. The prediction is recorded in testing_labels_pred.txt.
Please follow the link to access to the data. 

https://drive.google.com/open?id=1576O4NZbU4PJ4gw1YI11n4DbTAzNCIIj

# Data Preparation

Data preparation is the first step in the process of model development. This step was implemented in the Python programming language. Both “training_docs.txt” and “testing_docs.txt” are preprocessed using this technique. Python library used: re, nltk, pandas, multiprocessing, sklearn.feature_extraction.text Python code file: “Document preprocessing.ipynb”

Output file: “corpus2.csv” and “test 2.csv”

Data preparation steps are recorded below.
- Remove stop words
- Remove character “TEXT” at the beginning of the content
- Tokenise words by using regular expression with pattern r"\w+(?:[-.@']\w+)*"
- Lemmatise words
- Remove a word if the length of this word is less than 3
- Concatenate all remaining tokens into “nsw_token”, which will be used for feature selection in next stages
- Produce clean data input files under the names “corpus2.csv” and “test 2.csv”

# Features Selection

Library used: h2o (version 3.20.0.10) Example output: w2v_e30_v200_w30_f0
In the beginning, we considered two directions for feature selection using TF-IDF and using word embedding. After testing several models, we realised the embedding feature outperforms TF-IDF in this prediction task. The library h2o with the function h2o.word2vec supported our feature selection in all models.

From the H20 documentation, we try the following h2o.word2vec setting to produce features.

- Epochs: Specifies the number of training iterations to run.
- vec_size: Specifies the size of word vectors.
- window_size: This specifies the size of the context window around a specific word.
- min_word_freq: Specifies an integer for the minimum word frequency. Word2vec will discard words that appear less than this number of times.
Reference: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/word2vec.html

We extracted the features used for one of our best models as an example of features selection as an H2O object named “w2v_e30_v200_w30_f0”. This object has the following setting:
- epochs = 30
- Vector size = 200
- Window size = 30
- min_word_freq = 0

# Model selection
This section will describe the models that we used and assessed to ensemble the final result of the test set. In general, four types of model have been implemented: linear-based models, support vector machine models, Naïve Bayes and tree-based models. The best results will be generated from each model family and ensembled by a voting mechanism to achieve the final prediction. Furthermore, most of the models we used are from the h2o and liquidSVM packages because of their scalability, parallelisability, and computing resource optimisation. For detailed confusion matrix of each model, please refer to the appendix

*Please find the code of models in R_code folder*

## Naive Bayes (NB)
Model description: The first simple model built for analysing the input features is the Naïve Bayes with 5-folds CV. From this model, we will test and discuss more complex models to improve the accuracy in the following sections.
- Model description: H2O 
- Number of CV: 5
- Accuracy: 66.1%

## Generalised Linear Model (GLM)
Model description: In this model family, the multinomial family of the GLM algorithm of the h2o library is applied to 100% training data with 5-folds cross-validation. The average accuracy achieved is 75.8%, and the average MSE is 0.233.
- Model description: H2O 
- Number of CV: 5
- Accuracy: 75.8%

## Support Vector Machine (SVM)
- Model description: liquidSVM, H2O 
- Number of CV: 5
- Accuracy: 77.15%

## Auto Machine Learning H2O
Besides the above-mentioned traditional methods, we also applied function automl of the h2o library to generate the leaderboard of best algorithms used for this dataset. Finally, we chose the stacked ensemble model and distributed random forest model to make predictions.
For more information, please follow the documentation link: http://docs.h2o.ai/h2o/latest-stable/h2o- docs/automl.html
- Stacked ensemble accuracy: 76.1%
- Distributed random forest: 75.2%

## Ensemble

Ensemble models can gain advantages by reducing the variances of each classifiers. We applied a weighted voting technique for the ensemble to produce the final results. Each model is assigned a different vote. A ranking is first determined by highest CV accuracy and less overfitting. We allocated the better model (i.e., the model with the higher rank) with more votes and gave the underperforming model less votes. 

The R script used to ensemble result was written in “final ensemble.R” in the R_code folder. The final prediction is calculated by getting the prediction of each model and multiplying it by the number of votes for that specific model. The prediction with the highest frequency is chosen.

# Predicting result
We used the output result of the ensemble as our final submission. Our submission includes the following files:
- “testing labels pred.txt”: final prediction
- “Ensemble member.zip”: prediction of each member of the ensemble
- “Rcode.zip”: R codes used for predicting results
