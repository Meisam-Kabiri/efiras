# For Gemini (Google)
# pip install -U google-genai
from google import genai
from google.oauth2 import service_account

# 1. Path to your JSON key
KEY_PATH = "secrets/gcp_vertex-service-account-key.json"

# 2. Add the SCOPE here (This fixes the 'invalid_scope' error)
credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# 3. Initialize the Client
client = genai.Client(
    vertexai=True,
    project="efiras-480916",
    location="us-central1",
    credentials=credentials,
)

# 4. Use Gemini 2.5 Flash
# (Ensure you enabled Gemini 2.x in the Vertex AI Model Garden!)
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Write a punchy intro for a cool AI app."
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
