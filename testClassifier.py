import pandas as pd
import numpy as np
# import pickle

from keras.engine.saving import load_model
from keras.preprocessing.text import Tokenizer
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets as skds
from pathlib import Path

## for reproducablity
np.random.seed(444)

## Source file directory
path_train = "20news-bydate/20news-bydate-train"

files_train = skds.load_files(path_train, load_content=False)
label_index = files_train.target
label_names = files_train.target_names
labelled_files = files_train.filenames

data_tags = ['filename', 'category', 'news']
data_list = []

i = 0
for f in labelled_files:
    data_list.append((f,label_names[label_index[i]],Path(f).read_text(encoding='iso-8859-15')))
    i+=1

# We have training data available as dictionary filename, category, data
data = pd.DataFrame.from_records(data_list, columns=data_tags)
del data_list

# 20 news groups
num_labels = 20
vocab_size = 15000
batch_size = 100

# lets take 80% data as training and remaining 20% for test.
train_size = int(len(data) * .8)

train_posts = data['news'][:train_size]
train_tags = data['category'][:train_size]
# train_files_names = data['filename'][:train_size]

test_posts = data['news'][train_size:]
test_tags = data['category'][train_size:]
test_files_names = data['filename'][train_size:]

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)

# x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_tags)

# y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

model = load_model('my_model.h5')
score = model.evaluate(x_test,y_test, batch_size=batch_size, verbose=1)
print('test accuracy :', score[1])

test_label = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = test_label[np.argmax(prediction[0])]
    print(test_files_names.iloc[i])
    print('Actual label:', test_tags.iloc[i])
    print('Predicted label:', predicted_label)