import requests
import json


def query_llm(prompt):
    system_prompt = """You are an assistant that responds with JSON output containing function calls.
    Available functions:
    - gesture_control(command: str): Controls gesture system
    - system_command(action: str): Performs system actions
    
    Respond ONLY in this JSON format:
    {"function": "function_name", "parameters": {"param1": "value1"}}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma2:2b",
            "prompt": prompt,
            "system": system_prompt,
            "format": "json",
            "stream": False,
        },
    )

    try:
        return json.loads(response.json()["response"])
    except:
        return {"error": "Invalid response"}
