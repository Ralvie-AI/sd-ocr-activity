import os
import rapidocr

# Get the directory where rapidocr is installed
rapidocr_dir = os.path.dirname(rapidocr.__file__)

block_cipher = None

a = Analysis(['sd_ocr_activity/__main__.py'],
             pathex=[],
             binaries=None,
             datas=[(rapidocr_dir, 'rapidocr')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='sd-ocr-activity',
          contents_directory=".",
          debug=False,
          strip=False,
          upx=True,
          console=True )
          
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='sd-ocr-activity')