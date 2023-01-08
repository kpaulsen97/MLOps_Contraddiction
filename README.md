# MLOPS project description - Detecting contradiction and entailment in multilingual text

Beatrice Costanza Marrano s213290 \
Kenneth Paulsen s213291 

### Overall goal of the project
The goal of the project is to use natural language processing to solve a classification task of predicting whether two sentences entail, contradict each other, or are unrelated.
### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometric)
Since we chose a natural language processing problem, we plan to use the [Transformers](https://github.com/huggingface/transformers) framework.
### How do you intend to include the framework into your project
We are going to utilize the most appropriate NLP pre-trained model for the task, and train it further to improve accuracy. We have already identified XLMRobertaForSequenceClassification as a possible transformer. 
### What data are you going to run on
We are using the Kaggle competition [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) data. Each sample in the train set has the following information: a unique identifier, a premise, an hypothesis, the language and a target value whether the two sentences are entailed, unrelated or contradictory. 
