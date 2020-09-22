#fist neural network with keras tutorial
from numpy import argmax
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('/Users/nickmartin/projects/python/machine-learning-data/data/pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: {:02f}'.format(accuracy*100))

# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype('int32')

# summarize the first 5 cases
for i in range(5):
    print('{} => {} (expected {})'.format(X[i].tolist(), predictions[i], y[i]))

