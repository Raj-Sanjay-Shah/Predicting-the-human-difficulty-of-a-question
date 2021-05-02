# Predicting-the-human-difficulty-of-a-question
Creating a classifier that can distinguish the final line of a high school question from the final line of a college/high school questions.

I have used four models to test out the accuracy for the classification of the last line of the different type of questions.
1. BERT
2. ALBERT
3. Graph Convolutional Neural Network (Transductive nature but good for semi-supervised learning)
4. GCN - Cheby (Graph Convolutional Networks using Chebyshev Polynomials)

## Requirements
1. pip3 install -r requirements.txt
2. Download questions from [here](https://www.google.com/url?q=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpinafore-us-west-2%2Fqanta-jmlr-datasets%2Fqanta.train.2018.04.18.json&sa=D&sntz=1&usg=AFQjCNGf7EtqkO16UWbMx_eeAexvvoIXxw) and save as 'qanta.train.json'. Users can also change the value of the variable 'questions_file' in test_queries.py to the correct path.
3. Download documents from [here](https://www.google.com/url?q=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fpinafore-us-west-2%2Fqanta-jmlr-datasets%2Fwikipedia%2Fwiki_lookup.json&sa=D&sntz=1&usg=AFQjCNFJ_cCrB0wkRniaZ9yRWg7dvBslMw) and save as 'wiki_lookup.json'. Users can also change the value of the variable 'file_name_documents' in Index_Creation_code.py to the correct path.

## Steps to run the code:
1. Install all the requirements in the file requirements.txt by using the above code.
2. python run.py

## Results
| Model | Test Accuracy |
| :---:|     :---:      | 
|BERT| 69.8|
|ALBERT|64.4|
|GCN|54.27|
|GCN-Cheby|59.0|
