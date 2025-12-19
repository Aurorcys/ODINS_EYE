import numpy as np
import pandas as pd
np.random.seed(42)
returns = np.random.normal(0.001, 0.005, 252)
returns[50:100] += 0.02  # Add a lucky streak
pd.DataFrame({'returns': returns}).to_csv('perfect_strategy.csv')