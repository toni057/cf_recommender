"""
File contains classes:

BaselinePredictor:
    Implements the baseline predictor.


@author: toni
"""

from Base import Base

import pandas as pd
import numpy as np

from numpy.linalg import norm
from scipy.sparse import csr_matrix

from progressbar import ProgressBar, Percentage, Bar, ETA



class BaselinePredictor(Base):
    
    db = "db.sqlite3"
    
    
    def __init__(self, df = None, transform = False):
        if (df is not None):
            self.df = df
            
            if (transform == True):
                self.transformData()
    
    
    def fit_baseline(self, d = 1/5):
        """ 
        Fits the baseline predictor of the form:
            
            r_b = r_mean + b_u + b_i
        
        where:
            r_mean  : average rating among all users and items
            b_u     : user preference coefficient 
            b_i     : item preference coefficient 
        
        Parameters:
            d : floating point
                Dampening (regularization) coefficient.
        
        
        Solves the system ``min ||r - (r_mean + b_u + b_i)||^2 + d^2 ||b_u||^2 + d^2 ||b_i||^2``, for all pairs (u,i)
        
        The problem is translated to the system ``min ||Ax - b||^2 + d^2 ||x||^2``.
        """
        
        #n = self.df.shape[0]                                                                    # number of unique ratings
        self.r_mean = np.mean(self.df["rating"])                                                # mean rating
        
        # system matrices
        row = np.array([i for j in range(2) for i in range(self.n_rat)])                                 # row indices 1..n, 1..n
        col = np.array(self.df["item ind"].append(self.n_it + self.df["user ind"]))      # column indices
        data = np.array([1 for i in range(2 * self.n_rat)])                                                # ones
        
        A_mat = csr_matrix((data,(row,col)), shape = (self.n_rat, self.n_us + self.n_it))    # the matrix
        b_mat = np.array(self.df["rating"]) - self.r_mean                                       # b matrix
        
        
        x = lsqr(A_mat, b_mat, damp = d)[0]
        
        # b_u and b_i hold baseline factors for each user and item
        #self.b_u = pd.DataFrame({"user ind": np.array(range(self.n_us)), "b_u": x[self.n_it:]}, 
        #                        columns = ["user ind", "b_u"])
        #self.b_i = pd.DataFrame({"item ind": np.array(range(self.n_it)), "b_i": x[:self.n_it]}, 
        #                        columns = ["item ind", "b_i"])
        
        self.df_user["b_u"] = x[self.n_it:]
        self.df_item["b_i"] = x[:self.n_it]
        
        
        
    def evalBaseline(self, df = None):
        """ 
        Evaluates the baseline predictor.
        """
        
        if (df is None):
            self.r_b = self.df.merge(self.df_user[["user ind", "b_u"]], on = "user ind")
            self.r_b = self.r_b.merge(self.df_item[["item ind", "b_i"]], on = "item ind")
            self.r_b["baseline"] = self.r_mean + self.r_b["b_u"] + self.r_b["b_i"]
            
        
            return self.r_b[["user id", "item id", "baseline"]]
            
        else:
            df = df.merge(self.df_user, on = "user id").merge(self.df_item, on = "item id")
            df["baseline"] = self.r_mean + df["b_u"] + df["b_i"]
            
            # clip the score to the interval [1, 5]
            df["baseline"] = np.minimum(np.maximum(df["baseline"], 1), 5)
            
            return df[["user id", "item id", "baseline"]]

        
    def simMatrix(self, d = 1/5):
        """
        Item - item similarity matrix
        """
            
        self.fit_baseline(d)
        self.evalBaseline()
        
        
        df_mat = np.array(self.df[["user ind", "item ind", "rating"]].merge(self.r_b, on = ["user ind", "item ind"]))
        df_ind = df_mat[:,:2].astype(int)
        df_rat = df_mat[:,2] - df_mat[:,3]
        
        
        self.M = np.zeros((self.n_us, self.n_it))
        
        
        widgets = ['Test: ', Percentage(), ' ', Bar("#"), ' ', ETA()]
        pbar = ProgressBar(widgets = widgets, maxval = self.n_us)
        pbar.start()
        
        for us in self.user_ind:
            it = df_ind[np.where(df_ind[:,0] == us)[0], 1]
            rat1 = df_rat[np.where(df_ind[:,0] == us)[0]]
            self.M[us,it] = rat1
            
            pbar.update(us)
        
        pbar.finish()
        
        #self.M = self.UI.toarray()
        pbar = ProgressBar(widgets = widgets, maxval = self.n_it * (self.n_it - 1) / 2)
        pbar.start()
        
        self.S = np.empty((self.n_it, self.n_it)) * np.nan
            
        for i1 in range(self.n_it):
            # self.S[i1,i1] = 1
            x1 = self.M[:,i1]
            for i2 in range(i1+1,self.n_it):
                x2 = self.M[:,i2]
                I = np.logical_and(x1, x2)
                if (len(I) > 1):
                    self.S[i1,i2] = self.S[i2,i1] = Sim.cos2(x1.T[I], self.M[:,i2].T[I])
            
            pbar.update((self.n_it)*(i1+1) - (i1+2)*(i1+1)/2)
        
        pbar.finish()
        
        return self.S
    
    
    def storeNPY_S(self, f = "S.npy"):
        """
        Stores the S matrix in numpy format.
        """
        np.save(f, self.S)

        return
        
    def loadNPY_S(self, f = "S.npy"):
        """
        Loads the S matrix from numpy format.
        """
        self.S = np.load(f)
        
        return
    
    def storeNPY_UI(self, f = "UI.npy"):
        """
        Stores the User-Item matrix in numpy format.
        """
        np.save(f, self.UI)

        return
        
    def loadNPY_R(self, f = "UI.npy"):
        """
        Loads the User-Item matrix from numpy format.
        """
        self.UI = np.load(f)
        
        return
        
    
    def storeSQLite_S(self):
        
        m = [(i, j, self.S[i,j]) for i in range(self.n_it) for j in range(self.n_it) ]

        con = sqlite3.connect(self.db)
        con.execute("drop table if exists S;")
        con.execute("create table S (i1 integer, i2 integer, s float);")
        
        m = ([(i, j, self.S[i,j]) for i in range(self.n_it) for j in range(self.n_it)])
        
        con.executemany("insert into S values (?, ?, ?)", (m))
        con.commit()
        
        con.close()
    
    
    def loadSQLite_S(self):
        con = sqlite3.connect(self.db)
        
        cur = con.cursor()
        cur.execute("Select * from S;")
        m = cur.fetchall()
        """
        Ubaciti pretvaranje iz liste u matricu
        """
        cur.close()
        con.close()
        
        
