import numpy as np
class Scratch_Logistic_Regression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta_j  = None
        
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))    
    
    def fit(self, x_train, y_train):
        m = x_train.shape[0]
        
        bias_column = np.ones((m, 1))
        x_train_with_bias = np.column_stack((bias_column, x_train))
        
        n = x_train_with_bias.shape[1]
        self.theta_j = np.zeros(n)
        
        for i in range(self.iterations):
            y_predicted = self.sigmoid(self.theta_j @ x_train_with_bias.T)
            
            errors = y_train - y_predicted
            gradient = errors.T @ x_train_with_bias
            
            self.theta_j = self.theta_j + (self.learning_rate * gradient / m)
            
        
    def predict(self, X_Test):
        m_test = X_Test.shape[0]
        
        bias_column = np.ones((m_test, 1))
        X_Test_with_bias = np.column_stack((bias_column, X_Test))
        
        y_predicted = self.sigmoid(self.theta_j @ X_Test_with_bias.T)
        
        return (y_predicted >= 0.5).astype(int)


