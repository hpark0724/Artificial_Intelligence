import sys
import matplotlib.pyplot as plt
import numpy as np


if __name__ =="__main__":
    # 2
    dataset = open(sys.argv[1]) # the first argument as string

    Year = [] # Year which would be x-axis data
    Frozen_Days = [] # Number of Frozen Days which would be y-axis data

    next(dataset) # skip the first line of years and days in the csv dataset

    for line in dataset:
        year, days = line.split(",") # split the year and days by comma in the csv dataset
        Year.append(int(year)) # add the year data to the Year list
        Frozen_Days.append(int(days)) # add the days data to the Frozen_Days list

    plt.plot(Year, Frozen_Days) # plot the data with year as x-axis and days as y-axis

    plt.xlabel("Year") # label x-axis as "Year"
    plt.ylabel("Numer of Frozen Days") # label y-axis as "Number of Frozen Days"
    plt.savefig("plot.jpg") # save the plot as plot.jpg

    dataset.close() # close csv file

    # 3
    # 3.a
    xi = [] # create a list that will store the xi-> (1, each year in csv file (from 1st year - nth year)) 
            # data, i is in the range of (1,n)

    for year in Year: 
        xi.append([1,year]) # add the (1, each year in csv file) in the list of xi

    X = np.array(xi, dtype= np.int64) # convert xi list to the array for matrix calculation,
                                    # X.dtype needs to be int64
    print("Q3a:")
    print(X)

    # 3.b
    y = []  # create a list that will store the n number of frozen days data 

    for frozen_days in Frozen_Days: 
        y.append(frozen_days) # add each frozen days in the list of y

    Y = np.array(y, dtype = np.int64) # convert xi list to the array for matrix calculation,
                                    # Y.dtype needs to be int64
    print("Q3b:")
    print(Y)

    # 3.c
    Z = np.dot(X.transpose(), X) # compute the maxtrix product Z = X^T*X
    print("Q3c:")
    print(Z)

    # 3.d
    I = np.linalg.inv(Z) # inverse of the maxtrix product X^T * X
    print("Q3d:")
    print(I)

    # 3.e
    PI = np.dot(I, np.transpose(X)) # compute the maxtrix product, (X^T*X)^-1*X^T
    print("Q3e:")
    print(PI)

    # 3.f
    beta = np.dot(PI, Y) # compute beta matrix which is (beta0, beta1)^T
    print("Q3f:")
    print(beta)

    # 4
    x_test = 2022 # test data
    y_test = beta[0] + np.dot(beta[1], x_test) # test the frozen days in 2022 by
                                            # by plugging in the test data, 2022 to 
                                            # the expected linear regression of frozen days 
                                            # and year
    print("Q4: " + str(y_test)) # print the expected frozen days in 2022

    # 5
    if beta[1] > 0 :
        symbol = ">"
    elif beta[1] < 0 :
        symbol = "<"
    elif beta[1] == 0 :
        symbol = "="

    print("Q5a: " + symbol)

    print("Q5b: > means the frozen days will increase every year, < means The frozen days will decrease every year and" + 
        "= means the frozen days will remain constant")

    #6
    x0 = -beta[0]/beta[1] # calculate the year of when Lake Mendota will no longer freeze
    print("Q6a: " + str(x0))
    print("Q6b: While the linear regression expects around 2455 year," 
        +"Lake Mendota will no longer freeze, this prediction is"
        +"is not compelling since there would be lots of other factors"
        +"that influence the freeze days and a a negative frozen doesn't"
        +"make sense so the linear regression would not be suitable for "
        +"expecting frozen days of Lake Mendota")