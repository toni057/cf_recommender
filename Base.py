"""
File contains classes:

Base:
    Implements basic data loading and transformation to suitable formats.
    
    
@author: toni
"""


class Base:
    
    def __init__(self, df = None, transform = False):
        self.df = df
        
        if (transform == True):
            self.transformData()


    def readFromCSV(self, transform = False, UI_matrix = False):
        """
        Read data from csv files.
        
        Parameters:
            transform - whether to preprocess the data.
            UI_matrix - whether to create the USER-ITEM matrix.
        """
        #self.df = pd.read_csv('ml-10m/ratings.dat', sep = "::", names = ["user id", "item id", "rating", "timestamp" ])        
        self.df = pd.read_csv('ml-100k/u.data', sep = "\t", names = ["user id", "item id", "rating", "timestamp" ])
        
        if (transform == True):
            self.transformData()
    
    
    
    def readFromSQLite(self, transform = False, UI_matrix = False):
        """
        Read data from sqlite database.
        
        **** To be implemented.
        """
        pass
    
    
    def transformData(self, UI_matrix = False):
        """
        Ttransforms into suitable form data after loading.
        Independent from reading raw data, which can be held in various formats.
        """

        # all unique items
        self.items = np.sort(self.df["item id"].unique())
        self.item_ind = range(len(self.items))
        
        # all unique users
        self.users = np.sort(self.df["user id"].unique())
        self.user_ind = range(len(self.users))
        #self.user_mean = self.df.groupby(["user id"])["rating"].mean()
        #self.user_std = self.df.groupby(["user id"])["rating"].std()
        
        
        #preprocess - movies and users to indices
        self.df_item = pd.DataFrame({"item ind": self.item_ind, "item id": self.items})
        self.df_user = pd.DataFrame({"user ind": self.user_ind, "user id": self.users})
        #self.df_user = pd.DataFrame({"user ind": self.user_ind, "user id": self.users, "user mean": self.user_mean, "user std": self.user_std})
        
        self.df = self.df.merge(self.df_user)                  # merge user indices
        self.df = self.df.merge(self.df_item)                  # merge movie indices
        

        
        self.n_us = len(self.users)
        self.n_it = len(self.items)
        self.n_rat = len(self.df)
 
        # create the User - Item matrix R
        if (UI_matrix == True):
            # user - item matrix
            self.UI = lil_matrix((len(self.df_user), len(self.df_item)))

            for u in self.user_ind:
                i = self.df[["item ind", "rating"]][self.df["user ind"] == u]
                self.UI[u,i["item ind"]] = i["rating"]
                
            self.UI = csr_matrix(self.UI)
        
        self.userMean()
        self.itemMean()
    
    
    def userMean(self):
        """
        Calculates and returns mean ratings by user.
        
        """
        
        self.r_mean_us = np.array(self.df.groupby(["user ind"])[["rating"]].mean(), dtype = np.double)
        
        return self.r_mean_us
        
    def itemMean(self):
        """
        Calculates and returns mean ratings by item.
        """
        
        self.r_mean_it = np.array(self.df.groupby(["item ind"])[["rating"]].mean(), dtype = np.double)
        
        return self.r_mean_it

