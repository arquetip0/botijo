# /home/jack/botijo/testenv.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar el .env correcto desde /home/jack
dotenv_path = Path("/home/jack/.env")
load_dotenv(dotenv_path)

clave = os.getenv("ELEVENLABS_API_KEY")
print(f"Clave cargada correctamente: {clave}")
