SHELL := cmd.exe
PYTHON := .\venv3124\Scripts\python.exe

package: 
	call scripts\package.bat
	$(PYTHON) scripts\test.py

clean:
	rm -rf build dist
	rm -rf sd_ocr_activity/__pycache__

