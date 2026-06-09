import requests
import base64
import os
import json
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get configurations from environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") 

# GovAI API configuration
GOVAI_API_URL = os.getenv("GOVAI_API_URL")
GOVAI_API_KEY = os.getenv("GOVAI_API_KEY")
GOVAI_MODEL = os.getenv("GOVAI_MODEL")

class VisionProcessor:
    """
    A processor that uses vision models to analyze an image.
    Primary: GovAI LLM API (qwen25-vl-72b)
    Fallback: Local Ollama multimodal model
    It generates a caption and a list of descriptive tags.
    """

    def _analyze_with_govai(self, image_bytes):
        """
        Private helper method to encode an image and call the GovAI LLM API.
        
        Args:
            image_bytes (bytes): The raw bytes of the image in JPEG format.

        Returns:
            dict: The parsed JSON response from the GovAI model or None on failure.
        """
        if not GOVAI_API_KEY:
            print("GovAI API key not configured. Skipping GovAI and using fallback.")
            return None

        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{encoded_image}"

        system_prompt = """You are an expert image analysis AI. Your task is to produce detailed, accurate, and objective descriptions of images.
You must always respond with a single valid JSON object. Never include extra text outside the JSON.
Your descriptions should cover:
- Who is in the image: number of people, approximate age, gender, visible physical appearance.
- What they are wearing: clothing type, colors, accessories, headwear, eyewear, etc.
- What they are doing: actions, posture, gestures, expressions, interactions.
- Where the scene takes place: environment, setting, location clues (indoor/outdoor, vehicle, building, nature, etc.).
- What objects or notable elements are visible: vehicles, furniture, signs, animals, documents, technology, etc.
Be specific and descriptive. Do not be vague. Always respond in English only."""

        user_prompt = """Analyze this image carefully and respond with a JSON object containing:
1. "caption": A detailed, accurate description of the image. Describe the subjects (people, animals, objects), their appearance (clothing, colors, accessories), their actions or expressions, and the setting or environment. Be thorough and specific — aim for 2 to 4 sentences.
2. "tags": An array of 8-15 descriptive keywords covering people, clothing items, colors, objects, actions, setting, and mood."""

        print(f"Sending request to GovAI model: {GOVAI_MODEL}...")
        
        payload = {
            "model": GOVAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.5
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GOVAI_API_KEY}'
        }

        try:
            response = requests.post(GOVAI_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract the content from the OpenAI-style response format
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message_content = response_data['choices'][0].get('message', {}).get('content', '{}')
                
                # Try to parse as JSON, handling potential markdown code blocks
                json_string = message_content.strip()
                if json_string.startswith('```json'):
                    json_string = json_string[7:]
                if json_string.startswith('```'):
                    json_string = json_string[3:]
                if json_string.endswith('```'):
                    json_string = json_string[:-3]
                json_string = json_string.strip()
                
                try:
                    analysis_result = json.loads(json_string)
                    print("Successfully received and parsed response from GovAI.")
                    # Add the original image data to the result for display purposes
                    analysis_result["image_data"] = encoded_image
                    return analysis_result
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from GovAI response: {message_content}")
                    return None
            else:
                print(f"Error: Unexpected response format from GovAI: {response_data}")
                return None
                
        except requests.exceptions.Timeout:
            print("GovAI API request timed out.")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"GovAI API connection error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"GovAI API request failed: {e}")
            return None

    def _analyze_with_ollama(self, image_bytes):
        """
        Private helper method to encode an image and call the Ollama API.
        
        Args:
            image_bytes (bytes): The raw bytes of the image in JPEG format.

        Returns:
            dict: The parsed JSON response from the Ollama model or None on failure.
        """
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = """
        Analyze this image carefully and respond with a single JSON object containing two keys:
        1. "caption": A detailed, accurate description of the image. Describe the subjects (people, animals, or objects), their appearance (clothing, colors, accessories), their actions or expressions, and the environment or setting. Be specific and thorough — aim for 2 to 4 sentences.
        2. "tags": A JSON array of 8-15 descriptive keywords covering people, clothing items, colors, objects, actions, setting, and mood.
        Respond only with valid JSON. Always respond in English.
        """

        print(f"Sending request to Ollama model: {OLLAMA_MODEL}...")
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [encoded_image],
            "format": "json",
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        response_data = response.json()
        json_string = response_data.get("response", "{}")
        
        try:
            analysis_result = json.loads(json_string)
            print("Successfully received and parsed response from Ollama.")
            # Add the original image data to the result for display purposes
            analysis_result["image_data"] = encoded_image
            return analysis_result
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from model response: {json_string}")
            return None


    def process_image_from_bytes(self, image_bytes):
        """
        Processes an image directly from a byte stream.
        Tries GovAI API first, falls back to Ollama if GovAI fails.

        Args:
            image_bytes (bytes): The raw bytes of the image.

        Returns:
            dict: A dictionary with the analysis results, or None on failure.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                rgb_image = img.convert('RGB')
                buffered = io.BytesIO()
                rgb_image.save(buffered, format="JPEG")
                jpeg_bytes = buffered.getvalue()
            
            # Try GovAI API first (primary)
            result = self._analyze_with_govai(jpeg_bytes)
            if result:
                return result
            
            # Fallback to Ollama if GovAI fails
            print("GovAI failed or unavailable. Falling back to Ollama...")
            return self._analyze_with_ollama(jpeg_bytes)

        except Exception as e:
            print(f"An unexpected error occurred during byte processing: {e}")
            return None


    def process_image_from_path(self, image_path):
        """
        Processes an image from a local file path. This is used by the web UI.

        Args:
            image_path (str): The file path to the image.

        Returns:
            dict: A dictionary with the analysis results, or None on failure.
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return self.process_image_from_bytes(image_bytes)

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during path processing: {e}")
            return None

# Create a single instance to be used by the Flask app
processor_instance = VisionProcessor()