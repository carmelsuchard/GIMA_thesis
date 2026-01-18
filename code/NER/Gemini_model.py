from google import genai
from Gemini_api.py import API_key

# Make it an environment variable for security in real applications
GEMINI_API_KEY = API_key

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)