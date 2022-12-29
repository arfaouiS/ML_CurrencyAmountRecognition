import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


'''
Display label distribution in histogram
Args:
    dataframe containing data and labels
'''
def label_balanced(dataframe: pd.DataFrame):
    sns.countplot(data=dataframe, x="amount", hue="currency")
    return plt

'''
Display currency label distribution in pie plot
Args:
    dataframe containing data and labels
'''
def currency_balanced(data: pd.DataFrame):
        plt.subplot(2, 1, 1)
        x = data["currency"].value_counts().index
        y = data["currency"].value_counts().values
        percent = 100. * y / y.sum()
        data["currency"].value_counts().plot.pie(labeldistance=None)
        labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, percent)]
        plt.legend(labels, title="Categories", bbox_to_anchor=(0.85, 1.025), loc="upper left")
        plt.title("{} distribution".format("currency"))
        return plt

'''
Displays all data according to their labels
Args:
    dataframe containing data and labels
'''
def outliers_detection(dataframe: pd.DataFrame):
    images = []
    for i in range(len(dataframe)):
        images.append(dataframe.image[i])
    pca = PCA(n_components=2)
    data = pca.fit_transform(images)
    currency_label = dataframe['currency']
    amount_labels = dataframe['amount']
    labels = dataframe.apply(lambda x: str(x.currency) + ' - ' + str(x.amount), axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    for currency in np.unique(currency_label):
        i = np.where(currency_label == currency)
        ax[0, 0].scatter(data[:, 0][i], data[:, 1][i], label=currency)
    for amount in np.unique(amount_labels):
        i = np.where(amount_labels == amount)
        ax[0, 1].scatter(data[:, 0][i], data[:, 1][i], label=amount)
    for label in np.unique(labels):
        i = np.where(labels == label)
        ax[1, 0].scatter(data[:, 0][i], data[:, 1][i], label=label)
    ax[0, 0].set_title("Currencies of data")
    ax[0, 1].set_title("Amounts of data")
    ax[1, 0].set_title("Currencies and amounts of data")
    ax[1, 1].set_title("Data")
    ax[0, 0].legend(title='currency', loc='upper left', bbox_to_anchor=(1, 1))
    ax[0, 1].legend(title='amount', loc='upper left', bbox_to_anchor=(1, 1))
    ax[1, 0].legend(title='currency - amount', loc='upper left', bbox_to_anchor=(1, 1))
    ax[1, 1].scatter(data[:, 0], data[:, 1])
    return plt
