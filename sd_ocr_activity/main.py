import argparse
import logging 
from datetime import datetime
from pathlib import Path

from sd_ocr_activity.utils import setup_logging, get_log_dir
from sd_ocr_activity.ocr_activity import ActiveWindowOCRText
from sd_ocr_activity.systeminfo import get_system_info, print_system_info

logger = logging.getLogger(__name__)

SYSTEM_INFO_MARKER = "----- System Information -----"


def system_info_already_logged() -> bool:
    """Check whether system information has already been logged today."""

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(get_log_dir("sd-ocr-activity"))

    pattern = f"sd-ocr-activity_{date_str}.log*"

    for log_file in log_dir.glob(pattern):
        try:
            with log_file.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as file:
                if SYSTEM_INFO_MARKER in file.read():
                    return True

        except OSError as exc:
            logger.debug(
                "Unable to read log file %s: %s",
                log_file,
                exc,
            )

    return False
    
def main():

    parser = argparse.ArgumentParser(description="Imate to Text")
    parser.add_argument("--server_url", required=True, help="URL to update ocr text")
    parser.add_argument("--image_path", required=True, help="User ID for identification")
    parser.add_argument("--screenshot_id", type=int, default=0, help="Screenshot ID")

    args = parser.parse_args()

    # Set up logging
    setup_logging("sd-ocr-activity", log_file=True)

    if not system_info_already_logged():
        system_info = get_system_info()
        print_system_info(system_info)
        

    ActiveWindowOCRText(
        server_url=args.server_url,
        image_path=args.image_path,
        screenshot_id=args.screenshot_id
    ).run_ocr()    


if __name__ == '__main__':
    main()