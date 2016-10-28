import numpy as np
import csv
import sys

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient

    return theta

    def sigmoid(x):
        return 1 / (1 + np.exp(-x) + 1e-5)

    def Loss(Y, F):
        one = np.ones(Y.shape[0]).reshape(Y.shape[0], 1)
        L = (-(np.dot(Y.T, np.log(F)) + np.dot(((one-Y).T), np.log(one-F))))
        return L

# main code here
if __name__=='__main__':
    args = sys.argv
    if args[1] == 'train':
        data=[]
        x=[]
        y=[]
        with open('spam_data/spam_train.csv',newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        for i in range(len(data)):
            numdata = len(data[i])
            data[i] = [float(x) for x in data[i]]
            temp=[]
            for k in range(1,58):
                temp.append(data[i][k])
            x.append(temp)
            y.append(data[i][numdata-1])
        #4001
        x0=np.asarray(x)
        y0=np.asarray(y)
        print(y0)
        numIterations= 1000000
        # learning rate
        alpha = 0.000000001
        theta = np.ones(57)
        #do gradient descent
        theta = gradientDescent(x0, y0, theta, alpha, 57, numIterations)
        print(theta)

        #print model
        model = np.asarray(theta)
        np.savetxt("model.csv",model,delimiter=",")

    elif args[1] == 'test':
        data=[]
        m = []
        test = []
        y = []
        y.append(["id","label"])
        with open('model.csv',newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                m.append(row)

        for i in range(len(m)):
            m[i] = [float(x) for x in m[i]]

        with open('spam_data/spam_test.csv',newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)

        for i in range(len(data)):
            data[i] = [float(x) for x in data[i]]
            temp=[]
            for k in range(1,58):
                temp.append(data[i][k])
            test.append(temp)

        M = np.asarray(m)

        for i in range (0,600) :
            x = np.asarray(test[i])
            X = x.transpose()
            if np.dot(X, M) > 0.5 :
                y.append([i+1,1])
            elif np.dot(X, M) <= 0.5 :
                y.append([i+1,0])

        f = open("prediction.csv","w",newline='')
        w = csv.writer(f)
        w.writerows(y)
        f.close()
