from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
from functools import lru_cache

DetectorFactory.seed = 0

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = None
    target_lang: str = "en"
    
class DetectionRequest(BaseModel):
    text: str

@lru_cache(maxsize=10)  # Cache up to 10 models
def load_model(source_lang: str, target_lang: str):
    """Loads the MarianMT model and tokenizer for the specified language pair."""
    
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        return tokenizer, model
    
    except Exception as e:
        
        raise HTTPException(status_code=400, detail=f"Model not available for language pair: {source_lang}-{target_lang}. Error: {str(e)}")

def translate_text(text: str, tokenizer, model):
    """Translates text using the specified MarianMT model."""
    
    tokenized_text = tokenizer([text], return_tensors="pt")
    translated = model.generate(**tokenized_text)
    
    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.post("/translate")
def translate(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="must have an input text")
    
    try:
        # Detect source language if not provided
        source_lang = request.source_lang or detect(request.text)
        
        tokenizer, model = load_model(source_lang, request.target_lang)
        
        translation = translate_text(request.text, tokenizer, model)
    
        return {"source_text": request.text, "source_lang": source_lang, "translated_text": translation}
    
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(ex)}")


@app.post("/detect")
def detect_lang(request: DetectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Must have an input text")
    
    try:
        source_lang = detect(request.text)
        
        return {"source_text": request.text, "source_lang": source_lang}
    
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(ex)}")