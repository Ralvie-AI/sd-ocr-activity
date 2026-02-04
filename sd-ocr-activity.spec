from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = collect_data_files("rapidocr") + collect_data_files("openvino")
hiddenimports = collect_submodules("rapidocr") + collect_submodules("openvino")

a = Analysis(
    ['sd_ocr_activity/__main__.py'],
    pathex=[],
    binaries=None,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='sd-ocr-activity',
    debug=False,
    strip=False,
    upx=True,
    console=True
)