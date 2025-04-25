import requests
import sys
from pathlib import Path

def query_ollama(prompt, model="llama3", host="127.0.0.1", port=11434):
    """
    Send a prompt to the Ollama API and return the response.
    
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model to use, default is "llama3"
        host (str): The host running Ollama, default is "127.0.0.1"
        port (int): The port Ollama is running on, default is 11434
        
    Returns:
        str: The generated response from the model
    """
    url = f"http://{host}:{port}/api/generate"
    
    # Prepare the request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # Make the API call
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        result = response.json()
        
        # Extract the generated text
        generated_text = result.get("response", "")
        
        return generated_text
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None


# Example of how to use this in another script:
if __name__ == "__main__":
    # This code will only run if this file is executed directly
    # It won't run when the file is imported
    test_prompt = "Explain quantum computing in simple terms."
    result = query_ollama(test_prompt)
    print(result)
