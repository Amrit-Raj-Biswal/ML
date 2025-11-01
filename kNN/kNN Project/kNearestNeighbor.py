import operator
from collections import Counter

class kNearestNeighbors:
    def __init__(self, k):  # constructor/initializer
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print('Training Completed!')

    def predict(self, X_test):

        distance = {} # dictionary
        counter = 1 # will serve as key
        
        for i in self.X_train:
            distance[counter] = ((X_test[0][0] - i[0]) ** 2 + (X_test[0][1] - i[1]) ** 2) ** 1/2    # Euclidean distance for every point
            counter = counter + 1
        distance = sorted(distance.items(), key = operator.itemgetter(1))
        self.classify(distance=distance[:self.k])

    def classify(self, distance):
        label = []

        for i in distance:
            #print(self.y_train[i[0]])
            label.append(self.y_train[i[0]])

        return (Counter(label).most_common()[0][0])


        
