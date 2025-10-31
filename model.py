from fastapi import FastAPI 
import uvicorn 


app=FastAPI()

@app.get('/') 
def main() : 
    return {'message': 'I love you dutee'}

@app.get('/{name}')

def hello_name(name:str) : 
    return {'message': f'welcome to my world!,{name}'}