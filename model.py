from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

app = FastAPI()

# Define input schema
class RequestBody(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load dataset and train model
iris = load_iris()
X = iris.data
y = iris.target

clf = GaussianNB()
clf.fit(X, y)

# Endpoint for prediction
@app.post("/predict")
def predict(data: RequestBody):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predict class
    class_idx = clf.predict(test_data)[0]
    predicted_class = iris.target_names[class_idx]

    return {"class": predicted_class}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
