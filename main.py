import numpy as np



#data set
#X- hours of sleep & hours of studying, y- results
X = np.array(([3,5],[4,3],[8,10]), dtype = float)
y = np.array(([68],[55],[84]), dtype = float)

X = X/np.amax(X, axis = 0) #returns max values in axis numeber 0
y = y/100


class neural_net(object):
    def __init__(self):
        self.inputLayerSize = 2 #two columns of input data
        self.outputLayerSize = 1 #there will be one output
        self.hiddenLayerSize = 3 #only one hidden layer for this net but with 3 neurons

        #initializing weights
        #np.random.randn(t,k) - creates an array of a given size filled with standard distribution
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward_propagation(self, X):
        self.z2 = np.dot(X, self.W1) #output data multiplied by first weights
        self.a2 = self.sigmoid(self.z2) #in every neuron there is applied an activation function
        self.z3 = np.dot(self.a2, self.W2) #data from hidde layer multiplied by second weights
        yHat = self.sigmoid(self.z3) #creating result
        return yHat
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    #gradient of sigmoid function
    def sigmoid_derivative(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    def Cost_function(self, X, y):
        #computing cost for given x & y by using already existing weights
        self.yHat = self.forward_propagation(X) # here yHat is the answer given by forward_propagation() and we subtract it with the true value
        J = 0.5*sum((y - self.yHat)**2)
        return J
    def Cost_function_derivative(self, X, y):
        self.yHat = self.forward_propagation(X) #again yHat is the result of forward_propagation()
        delta1 = np.multiply(-(y-self.yHat), self.sigmoid_derivative(self.z3))
        derivative_J_W2 = np.dot(self.a2.T, delta1)

        delta2 = np.dot(delta1, self.W2.T)*self.sigmoid_derivative(self.z2)
        derivative_J_W1 = np.dot(X.T,delta2)

        return derivative_J_W1, derivative_J_W2
    #functions for better availbility of weights into other classes
    def get_params(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel())) #it takes arrays W1 and W2 and makes one 1d array params [a,b,c,d,......]
        return params
    def set_params(self,params):
        #what happens here we receive from get_params vector params [a,b,c,d,.....]
        #then this vector is being transformed twice for new W1 and W2 in the way we reshape this 1d vector into array of size inputlayersize*hiddenlayersize
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradient(self,X,y):
        #we take at first dJdW1 & dJdW2 as dreivatives of cost function
        #and then we turn these two arrays into on 1d vector
        dJdW1, dJdW2 = self.Cost_function_derivative(X,y)
        return np.concatenate((dJdW1.ravel(),dJdW1.ravel()))

def computeNumericalGradient(N,X,y):
    paramsInitial = N.get_params()
    numgrad = np.zeros(paramsInitial.shape) #creates array of 0 in shape of paramsInitial
    perturb = np.zeros(paramsInitial.shape)
    k = 1e-4

    for p in range(len(paramsInitial)):
        perturb[p] = k
        N.set_params(paramsInitial + perturb) #value of numerical gradient for f(x+epsilon)
        loss2 = N.Cost_function(X,y)

        N.set_params(paramsInitial - perturb)
        loss1 = N.Cost_function(X,y)

        #computing numerical gradient
        numgrad[p] = (loss2 - loss1)/(2*k)
        perturb[p] = 0

    N.set_params(paramsInitial)

    return numgrad

from scipy import optimize

class trainer(object):
    def __init__(self, NN):
        #local reference to already exisitng network
        self.NN = NN

    def callbackF(self, params):
        self.NN.set_params(params)
        self.J.append(self.NN.Cost_function(self.X, self.y))

    def costFunctionWrapper(self, params, X,y):
        self.NN.set_params(params)
        cost = self.NN.Cost_function(X,y)
        grad = self.NN.computeGradient(X,y)
        return cost, grad
    def train(self,X,y):
        self.X = X
        self.y = y

        self.J = [] #here I store costs

        params0 = self.NN.get_params()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X,y), options=options, callback=self.callbackF)

        self.NN.set_params(_res.x)
        self.optimizationResults = _res



NN = neural_net()
T = trainer(NN)
T.train(X,y)
cost1 = NN.Cost_function(X,y)
derivative_J_W1, derivative_J_W2 = NN.Cost_function_derivative(X,y)
scalar = 3
NN.W1 = NN.W1 + scalar*derivative_J_W1
NN.W2 = NN.W2 + scalar*derivative_J_W2
cost2 = NN.Cost_function(X,y)
answer = NN.forward_propagation(X)
print(answer)
#print(computeNumericalGradient(NN,X,y))
#print(NN.computeGradient(X,y))
#print(NN.Cost_function_derivative(X,y))





