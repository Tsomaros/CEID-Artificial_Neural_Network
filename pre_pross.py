import pandas as pd

data = ""
with open("dataset-HAR-PUC-Rio.csv") as file:
     data = file.read()
data = data.replace(',','.').replace(';',',')


x = open("new_dataset.csv","w")
x.writelines(data)
x.close()


df = pd.read_csv("new_dataset.csv")
df = df.drop(['user'], axis=1)
df['class'].replace(['sittingdown', 'standingup', 'standing', 'walking', 'sitting'], [1, 2, 3, 4, 5], inplace=True)
df['gender'].replace(['Woman', 'Man'], [0, 1], inplace=True)

df.drop(index=122076, inplace=True)

df.to_csv('new_dataset.csv', index=False, encoding="utf-8-sig")
