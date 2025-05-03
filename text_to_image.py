import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

import openai
import os
from typing import Optional
import base64
from PIL import Image
import io

class TextToImageModel:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_image(self, prompt: str, size: str = "1024x1024") -> str:
        """
        Generate an image using OpenAI's DALL-E model.
        Returns the path to the saved image.
        """
        try:
            response = openai.images.generate(
                prompt=prompt,
                n=1,
                size=size,
                response_format="b64_json"
            )
            
            # Decode base64 image
            image_data = base64.b64decode(response.data[0].b64_json)
            image = Image.open(io.BytesIO(image_data))
            
            # Save image
            output_dir = os.path.join(project_root, "assets", "generated_images")
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, f"generated_{hash(prompt)}.png")
            image.save(image_path)
            
            return image_path
            
        except Exception as e:
            raise Exception(f"Failed to generate image: {str(e)}")

# Example usage:
if __name__ == "__main__":
    model = TextToImageModel()
    prompt = "A professional, modern resume template with blue accents"
    path = model.generate_image(prompt)
    print(f"Image saved to {path}") 