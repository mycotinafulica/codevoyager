import os
from dotenv import load_dotenv

API_KEY_ENV_NAME = 'OPENAI_API_KEY'
def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv(API_KEY_ENV_NAME, '')

    if not api_key:
        print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
    elif not api_key.startswith("sk-proj-"):
        print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
    elif api_key.strip() != api_key:
        print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
    else:
        print("API key found and looks good so far!")

    return api_key