import numpy as np
# ________________price prediction for a 1650-square-foot house with 3 bedrooms using normal equation______________
data =np.loadtxt('ex1data2.txt', delimiter=',')
x=data[:,0:2]
y=data[:,2]
m=y.size
x=np.c_[(np.ones(m),x)]
xt=x.T
theta = np.linalg.pinv(xt.dot(x)).dot(xt).dot(y)

print('theta computed by normal equation:',theta)
predict = np.array([1, 1650, 3])
price = np.dot(predict, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price))
