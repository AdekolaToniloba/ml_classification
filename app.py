from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Any
from datetime import datetime
import main

app = FastAPI(title="ml_classification_api API")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "healthy"}
