# grammar_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from happytransformer import HappyTextToText, TTSettings

app = FastAPI()

# Load model once at startup (this is slow but happens only once)
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
settings = TTSettings(num_beams=5, min_length=1)
cefr_classifier = pipeline("text-classification", model="AbdulSami/bert-base-cased-cefr")
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

class Request(BaseModel):
    text: str

class Response(BaseModel):
    text: str

@app.post("/grammar-correct", response_model=Response)
async def grammar_correct(request: Request):
    try:
        result = happy_tt.generate_text(f"grammar: {request.text}", args=settings)
        return {"text": result.text}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}

@app.post("/cefr-vocab", response_model=Response)
async def cefr_vocab(request: Request):
    try:
        result = cefr_classifier(request.text)
        return {"text": result}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}

@app.post("/emotion-classifier", response_model=Response)
async def emotion(request: Request):
    try:
        result = emotion_classifier(request)
        return {"text": result[0][0]['label']}
    except Exception as e:
        return {"text": f"Error: {str(e)}"}
