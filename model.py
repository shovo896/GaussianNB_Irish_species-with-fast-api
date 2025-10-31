from fastapi import FastAPI 
import uvicorn 
from sklearn.datasets import load_iris 
from sklearn.naive_bayes import GaussianNB 
from pydantic import 
app=FastAPI()

class request_body(BaseModel) : 
    sepal_length : float 
    sepal_width  : float 
    petal_length : float 
    petal_width : float 

iris=load_iris()
#Getting our features and targets 

x=iris.data 
y=iris.target

clf=GaussianNB()
clf.fit(x,y)