# In this project we'll try to to use a linear regression model to predict housing prices

import requests
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Function to load in the dataset we will be using, assuming it's a csv file
def load_dataset(filename, separator=',', online=False):
    if online:
        # Data is stored online
        response = requests.get(filename)
        file_object = io.StringIO(response.content.decode('utf-8'))
    else:
        # Local file
        file_object = filename

    # Read the data using pandas
    raw_data = pd.read_csv(file_object, sep=separator)

    # Show the columns in the data that was read and return it
    raw_data.info()
    return raw_data


# Function to clean up the dataset
def cleanup(dataset):
    # Drop unused columns
    dataset.drop(dataset.columns[[0, 2, 3, 15, 17, 18]], axis=1, inplace=True)

    # Sanitize 'zindexvalue' (this is the index value Zillow.com allocated to a property)
    dataset['zindexvalue'] = dataset['zindexvalue'].str.replace(',', '')
    dataset['zindexvalue'] = pd.to_numeric(dataset['zindexvalue'])

    return dataset


def preprocess(dataset):
    # Let's add a 'price_per_sqft' column and start clustering all neighbourhoods in the dataset into three categories:
    # low price
    # high price/low frequency
    # high price/high frequency
    dataset['price_per_sqft'] = dataset['lastsoldprice'] / dataset['finishedsqft']
    freq = dataset.groupby('neighborhood').count()['address']
    mean = dataset.groupby('neighborhood').mean()['price_per_sqft']
    cluster = pd.concat([freq, mean], axis=1)
    cluster['neighborhood'] = cluster.index
    cluster.columns = ['freq', 'price_per_sqft', 'neighborhood']

    # Low price cluster
    cluster1 = cluster[cluster.price_per_sqft < 756]
    # High price/low frequency
    cluster_temp = cluster[cluster.price_per_sqft >= 756]
    cluster2 = cluster_temp[cluster_temp.freq < 123]
    # High price/high frequency
    cluster3 = cluster_temp[cluster_temp.freq >= 123]

    # Add a group column containing the cluster category
    def get_group(x):
        if x in cluster1.index:
            return 'low_price'
        elif x in cluster2.index:
            return 'high_price_low_freq'
        elif x in cluster3.index:
            return 'high_price_high_freq'
        else:
            return 'unknown'

    dataset['group'] = dataset.neighborhood.apply(get_group)

    # This now allows us to narrow which columns we keep for further analysis, as we've used these columns to group
    # our dataset into three different clusters
    dataset.drop(dataset.columns[[0, 4, 6, 7, 8, 13]], axis=1, inplace=True)
    dataset = dataset[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt', 'zindexvalue',
                       'group', 'lastsoldprice']]

    return dataset


# Main
if __name__ == '__main__':
    data = load_dataset('https://raw.githubusercontent.com/RuiChang123/Regression_for_house_price_estimation/master'
                        '/final_data.csv', online=True)
    data = cleanup(data)

    # Let's visualize some of the data we just loaded
    # Since we have longitude and latitude in our dataset, we can plot an actual map showing pricing across this map
    data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, figsize=(10, 7), c='lastsoldprice',
              cmap=plt.get_cmap('jet'), colorbar=True, sharex=False)

    # Now let's start predicting the 'lastsoldprice'
    # Let's first check the correlation between the columns in our dataset and the price
    corr_matrix = data.corr()
    print("\nCorrelation:\n{}".format(corr_matrix['lastsoldprice'].sort_values(ascending=False)))
    # The 'lastsoldprice' seems to have the biggest correlation to size ('finishedsqft') and number of bathrooms

    # Now let's pre-process out dataset and get rid of some no longer needed columns in the process
    data = preprocess(data)

    # Moving on to building our model by defining our input (X) and output (Y)
    X = data[['bathrooms', 'bedrooms', 'finishedsqft', 'totalrooms', 'usecode', 'yearbuilt', 'zindexvalue', 'group']]
    Y = data['lastsoldprice']

    # Create dummies for 'usecode' and 'group' (string values)
    n = pd.get_dummies(data.group)
    X = pd.concat([X, n], axis=1)
    m = pd.get_dummies(data.usecode)
    X = pd.concat([X, m], axis=1)
    drops = ['group', 'usecode']
    X.drop(drops, inplace=True, axis=1)

    # Now let's train our data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # And apply a LR model by fitting it to our trained data
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # Printing the R squared value, this represents the value by which Y can be explained by X
    print('\n=== Linear Regression ===\nR squared: {:.2f}% of the variability in Y can be explained by X'
          .format(model_lr.score(X_test, y_test) * 100))

    # Calculate MSE, RMSE and MAE
    y_pred = model_lr.predict(X_test)
    lr_mse = metrics.mean_squared_error(y_pred, y_test)
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = metrics.mean_absolute_error(y_pred, y_test)

    print('RMSE: predicted price value came within $ {:.2f} of the real price'.format(lr_rmse))
    print('MAE: {:.2f}'.format(lr_mae))

    # Let's try a different model to see how that compares: Random Forest
    model_forest = RandomForestRegressor(random_state=42)
    model_forest.fit(X_train, y_train)
    print('\n=== Random Forest ===\nR squared: {:.2f}% of the variability in Y can be explained by X'
          .format(model_forest.score(X_test, y_test) * 100))

    y_pred = model_forest.predict(X_test)
    forest_mse = metrics.mean_squared_error(y_pred, y_test)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = metrics.mean_absolute_error(y_pred, y_test)

    print('RMSE: predicted price value came within $ {:.2f} of the real price'.format(forest_rmse))
    print('MAE: {:.2f}'.format(forest_mae))

    # Show all plotted graphs
    plt.show(block=True)

