import os 
import sys 
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def start_exe_(exec_cmd, process_name=None):
    logger.info(f"Starting module {exec_cmd}")
    if not isinstance(exec_cmd, list):
        exec_cmd = [exec_cmd]        
    logger.debug("Running: {}".format(exec_cmd))

    # Don't display a console window on Windows
    # See: https://github.com/ActivityWatch/activitywatch/issues/212
    startupinfo = None
    if sys.platform == "win32" or sys.platform == "cygwin":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
   

    # There is a very good reason stdout and stderr is not PIPE here
    # See: https://github.com/ActivityWatch/aw-server/issues/27
    _process = subprocess.Popen(
        exec_cmd, universal_newlines=True, startupinfo=startupinfo
    )

def start_exe(exec_cmd, timeout_sec=15):
    logger.info(f"Starting module {exec_cmd}")
    if not isinstance(exec_cmd, list):
        exec_cmd = [exec_cmd]

    logger.debug("Running: {}".format(exec_cmd))

    # Don't display a console window on Windows
    # See: https://github.com/ActivityWatch/activitywatch/issues/212
    startupinfo = None
    if sys.platform in ("win32", "cygwin"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        # Use the 'with' statement to ensure underlying handles are cleaned up even if exceptions occur
        with subprocess.Popen(
                exec_cmd,
                universal_newlines=True,
                startupinfo=startupinfo
        ) as proc:

            try:
                # Block and wait, with a timeout mechanism to prevent the process accumulation
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                # If the exe hangs, force kill it to prevent processes from piling up!
                logger.error(f"Task execution timed out ({timeout_sec}s)! Force cleaning up...")
                proc.kill()
                proc.wait()
    except Exception as e:
        logger.error(f"Unexpected error occurred while starting the process: {e}")


script_dir = Path(__file__).resolve().parent
ocr_exe = os.path.join(script_dir, "dist", "sd-ocr-activity", "sd-ocr-activity.exe")
img_file = os.path.join(script_dir, "test.png")

cmd = [ocr_exe, '--server_url', 'http://localhost:7600/screenshot/update_ocr_text', '--image_path',
      img_file, '--screenshot_id', '1']


start_exe(cmd)