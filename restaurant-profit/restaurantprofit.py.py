import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import LogNorm

#--------------------------------------plottingdata----------------------------------------------------

data=np.loadtxt("ex1data1.txt",delimiter=",")
x=data[:,0]
y=data[:,1]
m=y.size  #number of trainig example
plt.scatter(x, y, color='blue', marker='x')  
plt.title("Scatter plot of training data")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s'")



#------------------------------------------Batch_gradient_descent--------------------------------------------
X=np.c_[np.ones(m),data[:,0]]
theta=np.zeros(2)
iteration=1500
alpha=0.01
#step1_compute the cost
def computecost(x,y ,theta):
    m=y.size
    cost=0
    cost=np.sum((np.dot( x, theta)-y)**2)/(2*m)
    return cost

#step2_gradient_descent_algo-
def GD(x,y,theta ,alpha,iteration):
    m=y.size
    for i in range(iteration):
        prediction=np.dot(x,theta).flatten()
        error=prediction-y
        gradient=(np.dot(x.T,error))*(1/m)
        theta=theta-alpha*gradient 
    return theta 
theta =GD(X,y,theta ,alpha,iteration)

#---------------------------------plot the linear fit------------------------------------------------------------------------
plt.plot(X[:, 1], X.dot(theta), color='red', label='Linear Regression')
plt.show()
#-----------------predict values for population 35,000 and 70,0000-----------------
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

#-----------------------------visualize cost function-----------------------------------------------------------------------
theta0_val= np.linspace(-10,10,100)
theta1_val= np.linspace(-1,4,100)
xs,ys = np.meshgrid(theta0_val,theta1_val)  # building a grid of all possible coordinate pairs from two ranges.(2 1D array ==> 2 2D grid)
J_val=np.zeros(xs.shape)
for i in range(len(theta0_val)):
    for j in range(len(theta0_val)):
        t=np.array([theta0_val[i],theta1_val[j]])
        J_val[i][j]=computecost(X,y,t)
J_val=J_val.T
#3D_plot
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(xs, ys, J_val)
ax1.set_xlabel('θ₀')
ax1.set_ylabel('θ₁')
ax1.set_zlabel('Cost J(θ)')
ax1.set_title('Surface Plot of Cost Function')
#countour_plots
plt.figure(2)
lvls=np.logspace(-2,3,20) #make 20 countour levels
plt.contour(xs,ys,J_val ,levels=lvls,norm=LogNorm())
plt.plot(theta[0],theta[1],c='r',marker='x')
plt.show()