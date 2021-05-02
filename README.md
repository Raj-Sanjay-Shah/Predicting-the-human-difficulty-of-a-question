# Predicting-the-human-difficulty-of-a-question
Creating a classifier that can distinguish the final line of a high school question from the final line of a college/high school questions.

I have used four models to test out the accuracy for the classification of the last line of the different type of questions.
1. BERT
2. ALBERT
3. Graph Convolutional Neural Network (Transductive nature but good for semi-supervised learning)
4. GCN - Cheby (Graph Convolutional Networks using Chebyshev Polynomials)
5. Hybrid Model


| Model | Test Accuracy |
| :---:|     :---:      |
|BERT| 66.8|
|ALBERT|64.4|
|GCN|56.2|
|GCN-Cheby|58.6|
|Hybrid|XXX|
