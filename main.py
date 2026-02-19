from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from google import genai
from google.genai import types

# 1. Initialize your FastAPI app
app = FastAPI()

# 2. Allow your workmate to use the API from their own website or machine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize the GenAI client
# It will pull the "GEMINI_API_KEY" from Render's Environment Variables box!
client = genai.Client()

# Define the structure for incoming data
class ComplaintRequest(BaseModel):
    text: str

# 4. HOME ROUTE: This makes the link look professional
@app.get("/")
async def home():
    return {
        "status": "Barangay AI API is Live",
        "testing": "Go to /docs to test it interactively",
        "endpoint": "/api/classify (POST)"
    }

# 5. THE AI CLASSIFIER ENDPOINT
@app.post("/api/classify")
async def analyze_complaint_g3_flash(request: ComplaintRequest):
    prompt = f"""
    You are an AI dispatcher for a Philippine Barangay.
    Analyze the following complaint: "{request.text}"
    
    Categorize it strictly into one of these Incident Types:
    [Theft & Robbery, Physical Injury, Fire & Disaster, Medical Emergency, VAWC, Public Disturbance, General Incident]
    
    And categorize it strictly into one of these Urgency Levels:
    [Critical, High, Medium, Low]
    """
    
    try:
        # Call Gemini 3 Flash and force a JSON output
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "incident_type": {
                            "type": "STRING",
                            "description": "The classified incident type."
                        },
                        "urgency_level": {
                            "type": "STRING",
                            "description": "The classified urgency level."
                        }
                    },
                    "required": ["incident_type", "urgency_level"]
                },
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                )
            ),
        )
        
        # Returns clean JSON to your workmate's code
        return json.loads(response.text)
    
    except Exception as e:
        return {"error": str(e)}