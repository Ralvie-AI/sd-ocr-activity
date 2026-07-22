import os 
from pathlib import Path 

SCREENSHOT_FOLDER = os.path.join(os.environ['LOCALAPPDATA'], "Sundial", "Sundial", "Screenshots")

CERT = (
    Path(os.getenv("LOCALAPPDATA"))
    / "Sundial"
    / "Sundial"
    / "tls"
    / "localhost.crt"
)