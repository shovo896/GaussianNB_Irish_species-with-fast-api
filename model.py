from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# Initialize app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load dataset & train model
iris = load_iris()
X = iris.data
y = iris.target
clf = GaussianNB()
clf.fit(X, y)

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Predict route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    test_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    class_idx = clf.predict(test_data)[0]
    predicted_class = iris.target_names[class_idx]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": f"Predicted Class: {predicted_class.capitalize()}"
        }
    )

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
