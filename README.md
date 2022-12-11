# ai-readability
Repository for CS 229 Final Project.

## `code` directory
* Contains all the main code for the project.
- `CLEAR_Corpus_6.01.csv` contains a CSV of all raw data from the CLEAR Corpus dataset.
- `CLEAR_Corpus_6.01.xlsc` contains the data as an Excel sheet.
- `cefrDATA.json` contains the cefr values/ratings for all exceprts in the dataset.
- `cefr_bot.js` web scraper used to get CEFR values for text excerpt data. Used the [Puppeteer API](https://pptr.dev/api/).
- `data_preprocessing.py` contains all the functions and code used to pre-process the text excerpt data (removing stopwords, replacing contractions, vectorization, etc.). Also contains some functions for report data visualizations.
- `deep_learning.py` code used to create and train the NN with different data input formats.
- `imbalanced.py` contains functions called to use different resampling techniques for MPAA ratings. Used in mpaa_rating.py. 
- `linear_regression.py` code used to perform baseline and non-baseline linear regression on partial and full feature sets.
- `mpaa_rating.py` contains code for performing KNN, Naive Bayes, boosting, and resampling for MPAA ratings.
- `mpaa_rating_nn.py` and `nn.py` contain code for testing deep learning imports and libraries. These were not used in producing any results for the final project.
- `raw_data.py` contains code to access the raw data and text excerpt data. 
- `package-lock.json` and `package.json` are files used by the `cefr_bot.js` web scraper.
- `text_difficulty_baseline1.py` gets the cefr rating for a single text excerpt and was used for testing purposes during the implementation of `cefr_bot.js`. 
- `mpaa_bert_nn.ipynb` contains up to date code for BERT and application. Used the Base BERT model from [Transformers by Hugging Face](https://huggingface.co/docs/transformers/model_doc/bert).

## `images` directory
Contains plots and images generated using the code in the `code` directory of the project.

## `keras_data` directory
Contains train/val output and MSE for the training and tweaking phase of the project for the neural network. 

## `mpaa_data` directory
Contains train/val output and MSE for the training and tweaking phase of the project for predicting MPAA ratings with KNN/Boosting/Resampling/Naive Bayes. 