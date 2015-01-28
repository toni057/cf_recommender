"""
File contains classes:

LatentFactorVariable1:
    Implements the SVD approach.
    

@author: toni
"""

from BaselinePredictor import BaselinePredictor


class LatentFactorVariable1(BaselinePredictor):
    
    
    def __init__(self, df, D = 20):
        self.df = df
        self.D = D
    
    
    def LatentFactorVariable1(self, gamma = 0.01, D = 50, max_iter = 40, lambda_reg_b = 0.005, lambda_reg_LF = 0.01, rel_tol = 0.000001):
        """
        Fits a latent factor variable model that solves the system:
            min ||r - (r_mean + b_u + b_i - p * q)||^2 + d * ( ||p||^2 + ||q||^2 + ||b_u||^2 + ||b_i||^2 ), for all (u,i)
            
            where:
                r - user ratings of items, 
                r_mean - global rating mean
                b_u - baseline user coefficient
                b_i - baseline item coefficient
                p - latent variable user matrix
                q - latent variable item matrix
                d - dampening (regularization) coefficient
                
        Learning algorithm employed is stochastic gradient descent. 
        
        Parameters:
            d - dampening (regularization)
            gamma - stochastic gradient descent (SGD) learning parameter
            D - latent variable dimension
            maxiter - maximum number of iterations
            lambda_reg_b - regularization parameter for the baseline part
            lambda_reg_LF - regularization parameter for the latent factor part
        """
        
        # initialization
        
        self.D = D
        
        df_mat = np.array(self.df[["user ind", "item ind", "rating"]])
        self.r_mean = df_mat[:,2].mean()
        
        self.p = (np.random.randn(D, len(self.users))) * 0.1
        self.q = (np.random.randn(D, len(self.items))) * 0.1
        self.b_u = (np.random.randn(len(self.users))) * 0.0
        self.b_i = (np.random.randn(len(self.items))) * 0.0
        
        n = len(df_mat)
        
        print (self.p[0,0])
        print (self.q[0,0])
        
        i = 0                                # SGD iterator
        MSE = np.zeros(max_iter) * np.inf    # MSE error vector
        
        # print "Iteration        MSE         Elapsed time"
        print ("%10s %20s %20s" %("Iteration", "MSE", "Rel. tol. goal"))
        
        while i in range(max_iter):
            
            next_iter = False
            
            MSE[i] = 0
            order = np.random.permutation(n)  # permute the order of observations - for stoch. grad. desc.
            
            for j in order:
                
                us, it, rat = df_mat[j,:]
                
                e = ((rat -
                      self.r_mean - self.b_i[it] - self.b_u[us] - 
                      np.dot(self.q[:,it], self.p[:,us])))
                
                MSE[i] += e ** 2
                
                
                self.b_u[us] += gamma * (e - lambda_reg_b * self.b_u[us])
                self.b_i[it] += gamma * (e - lambda_reg_b * self.b_i[it])
                
                self.p[:,us] += gamma * (e * self.q[:,it] - lambda_reg_LF * self.p[:,us])
                self.q[:,it] += gamma * (e * self.p[:,us] - lambda_reg_LF * self.q[:,it])
                
            
            MSE[i] = np.sqrt(MSE[i] / n)
            
            if (i == 0):
                # if initial iteration then next
                next_iter = True
                rel_tol_it = np.inf
                
            elif (i > 0):
                # if not the initial iteration then adjust the learning parameter
                if (MSE[i] < MSE[i-1]):
                    # if error is sufficiently small (less than 1%) then break
                    rel_tol_it = abs(MSE[i] - MSE[i-1]) / min(MSE[i], MSE[i-1])
                    if (rel_tol_it < rel_tol):
                        break
                    
                    # learning rate schedule
                    gamma = gamma * 0.97
                    #gamma = 1/ (0.01 * log(2 + i))
                    
                    next_iter = True
                    
                else:
                    # else if error increases then reduce the learning parameter and
                    # switch back to the parameters from the previous step (and discard current)
                    gamma = gamma / 2
                    self.b_u = param_prev[0]
                    self.b_i = param_prev[1]
                    self.p   = param_prev[2]
                    self.q   = param_prev[3]
                    rel_tol_it = param_prev[4]
                    
            if (next_iter == True):
                # added to avoid duplication 
                # increase iteration here and save current valid parameters
                param_prev = np.array([self.b_u, self.b_i, self.p, self.q, rel_tol_it])             # parameters from last iteration
                print ("%10d %20.4f %20.4f"  %(i, MSE[i], rel_tol_it))
                i += 1 
                    
        self.df_user["b_u"] = self.b_u
        self.df_item["b_i"] = self.b_i
        
        return MSE
    
    
    def evaluate(self, df = None):
        
        if (df is None):
            pass
            
        else:
            
            #b_u = pd.DataFrame({"user ind": range(len(self.b_u))})
            #b_u["b_u"] = self.b_u
            
            #b_i = pd.DataFrame({"item ind": range(len(self.b_i))})
            #b_i["b_i"] = self.b_i
            
            
            df = df.merge(self.df_user, how = "left", on = "user id")
            df = df.merge(self.df_item, how = "left", on = "item id")
            
            
            
            df_mat = np.array(df[["user ind", "item ind"]])
            
            rating = np.zeros(len(df), dtype = np.double)
            
            for i in range(len(df_mat)):
                us, it = df_mat[i,:2]
                rating[i] = (self.r_mean + 
                                   self.b_u[us] + 
                                   self.b_i[it] + 
                                   np.dot(self.q[:,it], self.p[:,us]))
            
            rating = np.minimum(np.maximum(np.array(rating), 1), 5)
            
            df = pd.DataFrame(df_mat, columns = ["user ind", "item ind"])
            
            df["rating"] = rating
            df = df.merge(self.df_user, how = "left", on = "user ind")
            df = df.merge(self.df_item, how = "left", on = "item ind")
            
            
            return df[["user id", "item id", "rating"]]
        
        

 