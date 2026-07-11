$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Не найдено виртуальное окружение .venv"
}

& $Python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot "ScanNum.spec")
if ($LASTEXITCODE -ne 0) {
    throw "Сборка EXE завершилась с ошибкой"
}

$Archive = Join-Path $PSScriptRoot "dist\ScanNum-0.4.0-win64.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path (Join-Path $PSScriptRoot "dist\ScanNum\*") -DestinationPath $Archive

Write-Host "EXE: $PSScriptRoot\dist\ScanNum\ScanNum.exe"
Write-Host "Архив для переноса: $Archive"
