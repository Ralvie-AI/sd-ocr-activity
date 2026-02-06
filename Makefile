package:
	pyinstaller sd-ocr-activity.spec --clean --noconfirm 

clean:
	rm -rf build dist
	rm -rf sd_ocr_activity/__pycache__

