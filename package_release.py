from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
APP_DIRECTORY = DIST / "ScanNum"
ARCHIVE = DIST / "ScanNum-0.6.1-win64.zip"


def main() -> None:
    if not (APP_DIRECTORY / "ScanNum.exe").is_file():
        raise FileNotFoundError("Сначала необходимо собрать dist/ScanNum/ScanNum.exe")
    shutil.copy2(ROOT / "open_data.cmd", APP_DIRECTORY / "Открыть данные.cmd")
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", APP_DIRECTORY)
    print(f"Архив для переноса: {ARCHIVE}")


if __name__ == "__main__":
    main()
