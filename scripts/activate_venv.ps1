# Convenience activation script for the project-local virtual environment.
# Usage (PowerShell):
#   . .\scripts\activate_venv.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $ActivateScript)) {
    Write-Host "[!] .venv 还没创建，请先运行: py -3.11 -m venv .venv" -ForegroundColor Yellow
    exit 1
}

. $ActivateScript
Write-Host "[OK] 已激活 GR-movie-recommendation 的 .venv (Python 3.11)" -ForegroundColor Green
python --version
