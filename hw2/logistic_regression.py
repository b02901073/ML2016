import numpy as np
import csv
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = y - sigmoid(hypothesis) 

        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))

        gradient = np.dot(xTrans, -loss)

        theta = theta - alpha * gradient

    return theta

# main code here
if __name__=='__main__':
    args = sys.argv
    if args[1] == 'train':
        data=[]
        x=[]
        y=[]
        with open(args[2],newline='') as f:
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
        numIterations= 100000
        # learning rate
        alpha = 0.00000001
        theta = np.ones(57)
        #do gradient descent
        theta = logistic(x0, y0, theta, alpha, 57, numIterations)
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

        with open(args[2],newline='') as f:
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
            a = np.dot(X, M)
            if np.dot(X, M) > 0.5 :
                y.append([i+1,1])
            elif np.dot(X, M) <= 0.5 :
                y.append([i+1,0])

        f = open("prediction.csv","w",newline='')
        w = csv.writer(f)
        w.writerows(y)
        f.close()
