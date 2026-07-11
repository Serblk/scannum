$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Не найдено виртуальное окружение .venv"
}

& $Python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "ScanNum.spec")
if ($LASTEXITCODE -ne 0) {
    throw "Сборка EXE завершилась с ошибкой"
}

& $Python (Join-Path $PSScriptRoot "package_release.py")
if ($LASTEXITCODE -ne 0) {
    throw "Упаковка ZIP завершилась с ошибкой"
}

Write-Host "EXE: $PSScriptRoot\dist\ScanNum\ScanNum.exe"
