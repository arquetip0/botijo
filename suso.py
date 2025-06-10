import os
from dotenv import load_dotenv
import pathlib # Importa pathlib para una manipulación de rutas más robusta

# ... otras importaciones ...

if __name__ == "__main__":
    print(f"DEBUG: El script se está ejecutando desde: {os.path.abspath(__file__)}")
    script_dir = pathlib.Path(__file__).resolve().parent # Directorio del script: /home/jack/botijo
    
    # Ruta al .env en /home/jack/ (un nivel arriba del directorio 'botijo')
    dotenv_path_calculado = script_dir.parent / '.env' 
    # Esto se traduciría a /home/jack/.env

    print(f"DEBUG: Directorio del script es: {script_dir}")
    print(f"DEBUG: Buscando .env en el directorio padre: {dotenv_path_calculado}")
    print(f"DEBUG: ¿Existe el archivo .env en la ruta calculada? {dotenv_path_calculado.exists()}")
    print(f"DEBUG: ¿Es un archivo? {dotenv_path_calculado.is_file()}")

    if dotenv_path_calculado.exists() and dotenv_path_calculado.is_file():
        load_dotenv(dotenv_path=dotenv_path_calculado) # Especifica la ruta al archivo .env
        print(f"Variables de entorno cargadas desde: {dotenv_path_calculado}")
        # Para verificar si la clave específica se cargó:
        api_key_value = os.getenv('ELEVENLABS_API_KEY')
        if api_key_value:
            print(f"DEBUG: Valor de ELEVENLABS_API_KEY después de load_dotenv: {api_key_value[:5]}...") # Muestra solo los primeros 5 caracteres por seguridad
        else:
            print("DEBUG: ELEVENLABS_API_KEY NO se cargó desde .env")
    else:
        print(f"Advertencia: Archivo .env no encontrado en {dotenv_path_calculado}. Asegúrate de que las API keys estén disponibles como variables de entorno del sistema.")

    # Comprobación de la API Key de ElevenLabs al inicio (ya la tienes, pero ahora debería funcionar si .env se carga)
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("ERROR CRÍTICO: La variable de entorno ELEVENLABS_API_KEY no está configurada después de intentar cargar .env.")
        print("La síntesis de voz con ElevenLabs no funcionará.")
        # exit(1) # Considera salir si la API key es indispensable
    
    main()