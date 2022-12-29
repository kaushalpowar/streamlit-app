import pandas as pd

penguins = pd.read_csv("/Users/kaushal/Documents/Projects/penguin_app/penguin_cleaned2.csv")

df = penguins.copy()
target = 'Species'
encode=['sex','Island']



for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]


target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2 }


def target_encode(val):
    return target_mapper[val]

df['Species'] = df['Species'].apply(target_encode)

X = df.drop('Species', axis=1)
Y = df['Species']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,Y)

import pickle
pickle.dump(clf, open('penguins_clf.pkl','wb'))
