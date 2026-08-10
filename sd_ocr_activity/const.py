import os 
from pathlib import Path 

SCREENSHOT_FOLDER = os.path.join(os.environ['LOCALAPPDATA'], "Sundial", "Sundial", "Screenshots")

TLS_DIR = Path(os.getenv("LOCALAPPDATA")) / "Sundial" / "Sundial" / "tls"

CERT_FILE = TLS_DIR / "localhost.crt"