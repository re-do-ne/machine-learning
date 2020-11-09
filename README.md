# Machine Learning

A small project showcasing several python implementations of the following ML models:

## Supervised Learning

The [Linear Regression](linear_regression.py) script takes a pre-made dataset containing data from the [San Francisco home sales database](www.sfgate.com) and [Zillow](www.zillow.com).
It then applies a Linear Regression model from the sklearn library on this data to make predictions, and compares the results to the sales price in the dataset. As a comparison, it then does the same using the sklearn Random Forest model.

## Unsupervised Learning

The [K-Means clustering](k-model.py) script takes the iris dataset, available as a default sklearn datasets, and applies the K-Means clustering model to this dataset.
It will first plot the entire dataset as is, then moves on to plot the data clustered by label and finally plots all labels including their respective centers.

## Reinforcement Learning

The [Q-Learning](q-learning.py) script makes use of the [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment provided by the OpenAI gym toolkit.
It initializes and maintains a basic Q-Learning table, as described [here](http://mnemstudio.org/path-finding-q-learning-tutorial.htm), which is then used to find the best approach to keeping a pole upright. 
