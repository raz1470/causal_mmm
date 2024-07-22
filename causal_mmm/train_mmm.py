import cvxpy as cp
import numpy as np

# hyper-opt for adstock using array

# need to be careful with feature scaling + coefficient constraints (currently disabled) - is MaxABScaler better?
# sort hill function
# need default values for hill which make it linear - or option not to apply
# train test split
# hill params need to be array tuned in hyper-opt

# mlflow

# visualisation helper functions

# causal graph


class TrainMMM:
    '''
    Class to train a marketing mix model with Ridge regression and specific coefficient constraints.
    
    Attributes:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable.
        reg_alpha (float): Regularization parameter.
        positive_indices (np.ndarray): Indices of coefficients that should be non-negative.
        negative_indices (np.ndarray): Indices of coefficients that should be non-positive.
        adstock_rate (float): Decay rate for the adstock transformation.
        hill_alpha (float): Hill saturation parameter alpha.
        hill_n (float): Hill saturation parameter n.
        hill_K (float): Hill saturation parameter K.
        optimized_coef (np.ndarray): Optimized coefficients after training.
        X_contributions (np.ndarray): Contribution of each feature.
        yhat (np.ndarray): Predicted values.
        contributions (np.ndarray): Contribution of each feature.   
        r2_score (float): R squared of the model.
        X_mean (np.ndarray): Mean of each feature for scaling.
        X_std (np.ndarray): Standard deviation of each feature for scaling.
    '''
    
    def __init__(self, 
                 X: np.ndarray,
                 y: np.ndarray,
                 reg_alpha: float,
                 positive_indices: np.ndarray,
                 negative_indices: np.ndarray,
                 adstock_rate: float,
                 hill_alpha: float,
                 hill_gamma: float):
        '''
        Initialize the TrainMMM class.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target variable.
            reg_alpha (float): Regularization parameter.
            positive_indices (np.ndarray): Indices of coefficients that should be non-negative.
            negative_indices (np.ndarray): Indices of coefficients that should be non-positive.
            adstock_rate (float): Decay rate for the adstock transformation.
            hill_alpha (float): Hill saturation parameter alpha.
            hill_gamma (float): Hill saturation parameter alpha.            
            hill_n (float): Hill saturation parameter n.
            hill_K (float): Hill saturation parameter K.
        '''
        self.X = X
        self.y = y
        self.reg_alpha = reg_alpha
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.adstock_rate = adstock_rate
        self.hill_alpha = hill_alpha
        self.hill_gamma = hill_gamma

    def scale_features(self) -> np.ndarray:
        '''
        Scale features to have zero mean and unit variance.
        
        Returns:
            np.ndarray: Scaled feature matrix.
        '''
        #self.X_mean = np.mean(self.X, axis=0)
        #self.X_std = np.std(self.X, axis=0)
        #self.X_scaled = (self.X - self.X_mean) / self.X_std
        self.X_scaled = self.X
        
        return self.X_scaled

    def add_intercept(self, X: np.ndarray) -> np.ndarray:
        '''
        Add an intercept (bias) term to the feature matrix.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Feature matrix with intercept term.
        '''
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def transform_adstock(self, X: np.ndarray) -> np.ndarray:
        '''
        Apply the geometric adstock transformation to the feature matrix.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Transformed data.
        '''
        adstocked = np.zeros_like(X)
        adstocked[0, :] = X[0, :]
        for t in range(1, X.shape[0]):
            adstocked[t, :] = X[t, :] + self.adstock_rate * adstocked[t - 1, :]
        self.adstocked = adstocked
        return self.adstocked
    
    def transform_hill(self, X: np.ndarray) -> np.ndarray:
        '''
        Apply the Hill saturation transformation to the feature matrix.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Transformed data.
        '''
        #return 1 / (1 + (self.hill_gamma / X) ** self.hill_alpha)
        return X

    def transform_features(self) -> np.ndarray:
        '''
        Apply both adstock and Hill transformations to the feature matrix.

        Returns:
            np.ndarray: Transformed feature matrix.
        '''
        self.scale_features()
        self.transform_adstock(self.X_scaled)
        self.X_transformed = self.transform_hill(self.adstocked)
        
        return self.X_transformed
    
    def train(self) -> np.ndarray:
        '''
        Train the model using Ridge regression with specified constraints.

        Returns:
            np.ndarray: Optimized coefficients.
        '''
        self.transform_features()
        self.X_with_intercept = self.add_intercept(self.X_transformed)
        coef = cp.Variable(self.X_with_intercept.shape[1])
        ridge_penalty = cp.norm(coef, 2)
        objective = cp.Minimize(cp.sum_squares(self.X_with_intercept @ coef - self.y) + self.reg_alpha * ridge_penalty)
        self.constraints = [coef[i] >= 0 for i in self.positive_indices] + [coef[i] <= 0 for i in self.negative_indices]
        problem = cp.Problem(objective, self.constraints)
        problem.solve()
        
        self.status = problem.status
                 
        self.optimized_coef = coef.value 
        
        return self.optimized_coef
    
    def predict(self) -> np.ndarray:
        '''
        Predict target variable using the coefficients from training.

        Returns:
            np.ndarray: Predicted values.
        '''
        self.yhat = self.X_with_intercept @ self.optimized_coef
        
        return self.yhat

    def calculate_contributions(self) -> np.ndarray:
        '''
        Calculate the contribution of each feature.

        Returns:
            np.ndarray: Contribution of each feature.
        '''
        self.contributions = self.X_with_intercept * self.optimized_coef.reshape(1, -1)

        return self.contributions
    
    def calculate_r2_score(self) -> float:
        '''
        Calculate the R squared of the model.

        Returns:
            np.ndarray: R squared of the model.
        '''        
        ss_res = np.sum((self.y - self.yhat) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        self.r2_score = 1 - (ss_res / ss_tot)
        
        return self.r2_score