from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

load_dotenv()
client = InferenceClient(
    provider="wavespeed",
    api_key=os.environ["HF_TOKEN_READ"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "A cute alien creature in a spaceship",
    model="black-forest-labs/FLUX.1-dev",
)

# Save the generated image to a file
image.save("generated_image.png")
print("Image saved as 'generated_image.png'")