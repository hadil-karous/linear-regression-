import numpy as np
import matplotlib.pyplot as plt
"""Feature Normalization ( when feautures have different ranges ) : 
make all features have *mean(center of data)=0 
                            how ? : substracting the average
                       *standard deviation ( how spread are the values)=1 
                             how ?: divide by std dev
AIM: helps the gradient descent converge faster """
#___________________FEATURE_NORMALIZATION_CODE______________________________________________________________
data =np.loadtxt('ex1data2.txt', delimiter=',')
x=data[:,0:2]
y=data[:,2]
m=y.size
#compute mean and std
mu = np.mean(x,axis=0)
std_dev=np.std(x,axis=0)
#normalizing
x_norm=(x-mu)/std_dev
#__________________GRADIENT_DESCENT____________________________________________________________
x=np.c_[np.ones(m),x_norm]
theta=np.zeros(3)
alpha=0.03
iteration=400
#compute_cost
def compute_cost(x,y,theta):
    m=y.size
    cost=np.dot((np.dot(x,theta)-y).T ,(np.dot(x,theta)-y))/(2*m)
    return cost  
#gradientdescent
def gradient_descent(x,y,theta,alpha,iteration):
    m=y.size
    J_history=np.zeros(iteration)
    for i in range(iteration):
        prediction= x@theta
        error=prediction-y
        gradient=(x.T @ error )/m
        theta-=alpha*gradient
        J_history[i]=compute_cost(x,y,theta)
    
    return theta , J_history
theta , J_history = gradient_descent(x,y,theta,alpha,iteration)
#--------------- plotting the cost against the number of iteration (convergence graph)----------------------
plt.figure(1)
plt.plot(np.arange(iteration),J_history)
plt.xlabel('number of iteration')
plt.ylabel('cost J')
plt.show()
#----------------ESTIMATE THE PRICE OF A 1650 sq_ft ,3 bedr ------------------
predict=((np.array([1650,3])-mu)/std_dev)
predict=np.r_[(1,predict)]
price= predict @ theta 
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price))

