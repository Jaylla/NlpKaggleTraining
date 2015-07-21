# Text sentiment prediction
It's a code for Kaggle competition ["Bag of Words Meets Bags of Popcorn"](https://www.kaggle.com/c/word2vec-nlp-tutorial). Final score is 0.84588. 
Files BagOfWords.py and Word2Vec_AverageVectors.py contain code from tutorial with fiew modifications. And W2V_AverageTopWords.py contains code of my final submission. 

### The model description
The first step is to select the most significant words, using sklearn.feature_selection.SelectKBest metod (chi-squared test).
The second step is to count average vectors from Word2Vec representation of paragraph words (only words from first step are taken into account).
And finally model is trained using Linear Regression. 
Final score is 0.93