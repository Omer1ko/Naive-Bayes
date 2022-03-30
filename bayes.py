import numpy
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy import nan


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def bayeslearn(x_train: np.array, y_train: np.array):
    """
    :param x_train: 2D numpy array of size (m, d) containing the the training set. The training samples should be binarized
    :param y_train: numpy array of size (m, 1) containing the labels of the training set
    :return: a triple (allpos, ppos, pneg) the estimated conditional probabilities to use in the Bayes predictor
    """
    m = np.size(x_train)
    d = np.size(x_train[0])

    allpos = np.size(y_train[y_train==1]) / np.size(y_train)
    ppos = np.zeros(d)
    #ppos = ppos.reshape((d,1))
    pneg = np.zeros(d)
    #pneg = pneg.reshape((d,1))

    for i in range (np.size(x_train[0])):
        if(i==150):
            d=13
        column_i = x_train[:, i]
        tempPPOS = column_i[y_train==1]
        a = np.size(tempPPOS[tempPPOS==1])
        b = np.size(tempPPOS)
        ppos[i] = a / b


        tempPNEG = column_i[y_train==-1]
        pneg[i] = np.size(tempPNEG[tempPNEG==1]) / np.size(tempPNEG)

    return (allpos, ppos, pneg)

def bayespredict(allpos: float, ppos: np.array, pneg: np.array, x_test: np.array):
    """

    :param allpos: scalar between 0 and 1, indicating the fraction of positive labels in the training sample
    :param ppos: numpy array of size (d, 1) containing the empirical plug-in estimate of the positive conditional probabilities
    :param pneg: numpy array of size (d, 1) containing the empirical plug-in estimate of the negative conditional probabilities
    :param x_test: numpy array of size (n, d) containing the test samples
    :return: numpy array of size (n, 1) containing the predicted labels of the test samples
    """

    firstTerm = math.log2((allpos)/(1-allpos))
    coPPOS = np.ones(np.size(ppos))-ppos
    coPNEG = np.ones(np.size(pneg))-pneg
    ppos[ppos==0] = 1
    pneg[pneg==0] = 1
    coPPOS[coPPOS==0] = 1
    coPNEG[coPNEG==0] = 1
    secondTermMulRow = np.log2(ppos)-np.log2(pneg)
    thirdTermMulRow =  np.log2(coPPOS)-np.log2(coPNEG)
    #secondTermMulRow= np.nan_to_num(secondTermMulRow, 0)
    #thirdTermMulRow= np.nan_to_num(thirdTermMulRow, 0)
    thirdTermMulRow=np.reshape(thirdTermMulRow, (784,1))
    secondTermMulRow=np.reshape(secondTermMulRow, (784,1))


    predict = numpy.ones(np.shape(x_test)[0])
    predict = predict*firstTerm
    predict=np.reshape(predict, (np.shape(x_test)[0],1))

    predict = predict + x_test @ secondTermMulRow
    x_test1 = np.ones(np.shape(x_test))
    x_test1 = x_test - 1*x_test1
    x_test1 = x_test1 * -1

    predict = predict + x_test1 @ thirdTermMulRow
    predict = np.sign(predict)
    predict = np.reshape(predict, (np.size(predict),1))
    return predict






def simple_test():
    # load sample data from question 2, digits 3 and 5 (this is just an example code, don't forget the other part of
    # the question)
    data = np.load('mnist_all.npz')

    train3 = data['train3']
    train5 = data['train5']

    test3 = data['test3']
    test5 = data['test5']

    m = 500
    n = 50
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)
    x_test, y_test = gensmallm([test3, test5], [-1, 1], n)

    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    assert isinstance(ppos, np.ndarray) \
           and isinstance(pneg, np.ndarray), "ppos and pneg should be numpy arrays"

    assert 0 <= allpos <= 1, "allpos should be a float between 0 and 1"

    y_predict = bayespredict(allpos, ppos, pneg, x_test)

    assert isinstance(y_predict, np.ndarray), "The output of the function bayespredict should be numpy arrays"
    assert y_predict.shape == (n, 1), f"The output of bayespredict should be of size ({n}, 1)"
    y_test=np.reshape(y_test,(50,1))
    print(f"Prediction error = {np.mean(y_test != y_predict)}")


def Task2a():
    #2a
    data = np.load('mnist_all.npz')
    train3 = data['train3']
    train5 = data['train5']
    test3 = data['test3']
    test5 = data['test5']
    train0 = data['train0']
    train1 = data['train1']
    test0 = data['test0']
    test1 = data['test1']

    trainSize = np.array([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
    errorVector53 = np.zeros(10)
    errorVector10 = np.zeros(10)

    for i in range(10):
        m = trainSize[i]
        n = 1500
        x_train53, y_train53 = gensmallm([train3, train5], [-1, 1], m)
        x_test53, y_test53 = gensmallm([test3, test5], [-1, 1], n)

        threshold = 128
        x_train53 = np.where(x_train53 > threshold, 1, 0)
        x_test53 = np.where(x_test53 > threshold, 1, 0)
        allpos53, ppos53, pneg53 = bayeslearn(x_train53, y_train53)
        y_predict53 = bayespredict(allpos53, ppos53, pneg53, x_test53)
        y_test53 = np.reshape(y_test53, (np.size(y_test53), 1))
        errorVector53[i]=np.mean(y_test53 != y_predict53)

        x_train10, y_train10 = gensmallm([train0, train1], [-1, 1], m)
        x_test10, y_test10 = gensmallm([test0, test1], [-1, 1], n)

        threshold = 128
        x_train10 = np.where(x_train10 > threshold, 1, 0)
        x_test10 = np.where(x_test10 > threshold, 1, 0)
        allpos10, ppos10, pneg10 = bayeslearn(x_train10, y_train10)
        y_predict10 = bayespredict(allpos10, ppos10, pneg10, x_test10)
        y_test10 = np.reshape(y_test10, (np.size(y_test10), 1))
        errorVector10[i] = np.mean(y_test10 != y_predict10)

    p1 = plt.plot(trainSize, errorVector53,label="5 and 3")
    p2 = plt.plot(trainSize, errorVector10,label="0 and 1")
    plt.xlabel('Train Size')
    plt.ylabel('Error')
    plt.title('Test errors as a function of the sample size')
    plt.legend()
    plt.show()
    plt.clf()

def Task2c():
    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']

    m = 10000
    x_train10, y_train10 = gensmallm([train0, train1], [-1, 1], m)

    threshold = 128
    x_train10 = np.where(x_train10 > threshold, 1, 0)
    allpos10, ppos10, pneg10 = bayeslearn(x_train10, y_train10)
    ppos10=np.reshape(ppos10,(28,28))
    pneg10=np.reshape(pneg10,(28,28))
    plt.imshow(ppos10,cmap='hot')
    plt.title("PPOS Heat Map")
    plt.show()
    plt.clf()

    plt.imshow(pneg10,cmap='hot')
    plt.title("PNEG Heat Map")
    plt.show()
    plt.clf()


def Task2d():
    data = np.load('mnist_all.npz')

    train3 = data['train3']
    train5 = data['train5']
    test3 = data['test3']
    test5 = data['test5']

    m = 10000
    n = 1000
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)
    x_test, y_test = gensmallm([test3, test5], [-1, 1], n)
    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)
    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)


    y_predict = bayespredict(allpos, ppos, pneg, x_test)
    y_predictNew = bayespredict(0.75, ppos, pneg, x_test)


    y_test=np.reshape(y_test,(np.size(y_test),1))
    print("Used training size of 10000, and test size of ", n,'\n')

    #print(f"Original rediction error = {np.mean(y_test != y_predict)}")
    #print(f"New rediction error = {np.mean(y_test != y_predictNew)}",'\n')
    diffrences = y_predict-y_predictNew

    # If labels changed  from 1 to -1, diffrences array Value will be 2
    # If labels changed  from -1 to 1, diffrences array Value will be -2
    countChanged1 = np.size(diffrences[diffrences==2])
    countChanged2 = np.size(diffrences[diffrences==-2])
    print(countChanged1/m, "% of the labels changed  from 1 to -1")
    print(countChanged2/m, "% of the labels changed  from -1 to 1")












if __name__ == '__main__':
    Task2a()
    Task2c()
    Task2d()
