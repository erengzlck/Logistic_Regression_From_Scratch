import numpy as np
import pandas as pd

class Logistic_Regression():
    def __init__(self, distribution, method, learning_rate, epochs):
        self.epochs = epochs
        self.distribution = distribution
        self.learning_rate = learning_rate
        method = method.split('_')
        if method[0] == 'Full-Batch':
            self.method = 'Full-Batch'
            
        elif method[0] == 'Stochastic':
            self.method = 'Stochastic'
            
        elif method[0] == 'Mini-Batch':
            self.method = 'Mini-Batch'
            self.batch_size = int(method[1])
    
    def min_max(self, x):
        minmax = list()
        for i in range(len(x[0])):
            col_values = [row[i] for row in x]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    def normalize(self, x, minmax):
        for row in x:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    def split_normalize(self, filename, val_ratio, test_ratio):
        array = pd.read_csv(filename).values
        np.random.shuffle(array)
        x = array[:, :-1]
        y = array[:, -1]

        train_len = int(x.shape[0] * (1 - (val_ratio + test_ratio)))
        val_len = int(x.shape[0] * val_ratio)

        x_train = x[:train_len]
        x_val = x[train_len + 1: train_len + val_len + 1]
        x_test = x[train_len + val_len :]

        y_train = y[:train_len]
        y_val = y[train_len + 1: train_len + val_len + 1]
        y_test = y[train_len + val_len :]

        y_train = y_train.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        minmax = self.min_max(x_train)
        arrays_x = [x_train, x_val, x_test]
        for i in arrays_x:
            self.normalize(i, minmax)

        x_train = np.hstack((np.ones((x_train.shape[0],1)),x_train))
        x_val = np.hstack((np.ones((x_val.shape[0],1)),x_val))
        x_test = np.hstack((np.ones((x_test.shape[0],1)),x_test))


        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
        return x_train, y_train, x_val, y_val, x_test, y_test

    def calc_z(self, x, weights):
        return np.matmul(x, weights)

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0

        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def gradient(self, x, y, weights):
        y_hat = self.calc_y_hat(x, weights)
        grad = np.dot(x.transpose(), (y_hat - y))
        return grad

    def fit(self, X, y):
        accuracies = []
        if self.distribution == 'Gaussian':
            weights = np.random.normal(0,1,size = (X.shape[1], 1))
        elif self.distribution == 'Uniform':
            weights = np.random.uniform(size = (X.shape[1], 1))
        elif self.distribution == 'Zeros':
            weights = np.zeros((X.shape[1], 1))
        if self.method == 'Full-Batch':
            for itr in range(self.epochs):
                weights = weights - self.learning_rate * self.gradient(X, y, weights)
                self.weights = weights
                predicted = self.predict(self.x_val)
                ac = self.accuracy_score(self.y_val, predicted)
                accuracies.append(ac)
                self.accuracies = accuracies

            self.weights = weights
           
        elif self.method == 'Mini-Batch':
            for itr in range(self.epochs):
                mini_batches = self.create_mini_batches(X, y, batch_size = self.batch_size)
                for mini_batch in mini_batches:
                    X_mini, y_mini = mini_batch
                    weights = weights - self.learning_rate * self.gradient(X_mini, y_mini, weights)
                self.weights = weights
                predicted = self.predict(self.x_val)
                ac = self.accuracy_score(self.y_val, predicted)
                accuracies.append(ac)
                self.accuracies = accuracies

            self.weights = weights
            
        elif self.method == 'Stochastic':
            m = len(y)
            for itr in range(self.epochs):
                for i in range(m):
                    n = np.random.randint(0,m)
                    X_i = X[n, :].reshape(1, X.shape[1])
                    y_i = y[n, :].reshape(1, 1)
                    weights = weights - self.learning_rate * self.gradient(X_i, y_i, weights)
                self.weights = weights
                predicted = self.predict(self.x_val)
                ac = self.accuracy_score(self.y_val, predicted)
                accuracies.append(ac)
                self.accuracies = accuracies

            self.weights = weights

    def calc_y_hat(self, x, weights):
        a = np.dot(x, weights)
        return np.exp(a) / (1 + np.exp(a))


    def sigmoid(self, z):
        sig = []
        for i in range(len(z)):
            if z[i][0] >= 0:
                x = np.exp(-z[i][0])
                sig.append(1 / (1+x))
            else:
                x = np.exp(z[i][0])
                sig.append(x/ (1 + x))
        return sig

    def predict(self, x):
        z = self.calc_z(x, self.weights)
        results = []
        for i in self.sigmoid(z):
            if i > 0.5:
                results.append(1)
            else:
                results.append(0)
        results = np.array(results)
        results = results.reshape(-1,1)
        return results
    
    def confusion_matrix(self, true, predicted):
        labels = np.unique(predicted)
        matrix = np.zeros(shape=(len(labels),len(labels)))
        for i in range(len(labels)):
            for j in range(len(labels)):
                matrix[j,i] = np.sum((true == labels[i]) & (predicted == labels[j]))
        matrix = matrix.astype('int')
        self.matrix = matrix
        return matrix

    def accuracy_score(self, true, predicted):
        accuracy = np.sum(np.equal(true, predicted)) / len(true)
        return accuracy
    
    def f_metrics(self):
        precision = self.matrix[1][1] / (self.matrix[1][1] + self.matrix[1][0])
        recall = self.matrix[1][1] / (self.matrix[1][1] + self.matrix[0][1])
        fpr = self.matrix[1][0] / (self.matrix[1][0] + self.matrix[0][0])
        
        F_05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
        F_1 = (1 + 1**2) * (precision * recall) / ((1**2 * precision) + recall)
        F_2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
        
        return F_05, F_1, F_2, fpr
