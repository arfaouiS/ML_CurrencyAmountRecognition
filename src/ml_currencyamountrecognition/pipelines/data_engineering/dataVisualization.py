import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#TODO: correlation des données pour sélectionner les features
#TODO: Distribution des données : si une classe est prépondérrante l'algo va trop souvent prédire cette classe et avoir de bon résultat
#TODO: visualization des données pour voir les outliers
#https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html


def label_balanced(dataframe: pd.DataFrame):
    sns.countplot(data=dataframe, x="amount", hue="currency")
    return plt

'''
def currency_label_balanced(data: pd.DataFrame):
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