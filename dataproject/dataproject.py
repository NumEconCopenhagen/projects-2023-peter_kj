import numpy as np


def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df


def lin_reg(x,y):
    ''' This function takes two vectors as inputs. It then creates a linear regression. The outputs are the resulting constant, coefficient and predicted value.'''

    import numpy as np
    y = np.array(y) 
    x = np.array(x)
    mean_y = np.mean(y)
    mean_x = np.mean(x)

    n = len(y)
    

    numer = 0
    denom = 0

    for i in range(n):
        numer += (x[i] - mean_x) * (y[i] - mean_y)
        denom += (x[i] - mean_x) ** 2
    beta = numer / denom
    alpha = mean_y - (beta * mean_x)

    y_hat = alpha + beta * x

    return alpha, beta, y_hat