https://www.superdatascience.com/
11. GET THE DATASET
Identify independent variables to predict dependant variables
Country,Age,Salary - IV
Purchased - DV

12. Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
No libraries to be imported in R

13. Importing Dataset
dataset = pd.read_csv('Data.csv')
dataset = read.csv('Data.csv')

15. Missing Data
Remove the row OR
Insert MEAN
from sklearn.preprocessing import Imputer
(ctrl + I to get HELP)