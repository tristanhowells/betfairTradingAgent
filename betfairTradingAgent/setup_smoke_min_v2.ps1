
# setup_smoke_min_v2.ps1 - minimal PowerShell setup and smoke test (fixed Join-Path)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Err ($m){ Write-Host "[ERROR] $m" -ForegroundColor Red }

# Resolve project root (script in repo root or parent of sac_lstm_au_racing)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir

if (-not (Test-Path (Join-Path -Path $projectRoot -ChildPath 'requirements.txt'))) {
  $candidate = Join-Path -Path $projectRoot -ChildPath 'sac_lstm_au_racing'
  if (Test-Path (Join-Path -Path $candidate -ChildPath 'requirements.txt')) {
    $projectRoot = $candidate
  } else {
    Err "requirements.txt not found in '$projectRoot' or '$candidate'."
    exit 2
  }
}
Info "Project root: $projectRoot"
Set-Location $projectRoot

# Ensure Python
try {
  $py = & python -c "import sys; print(sys.version.split()[0])"
  Info "Python detected: $py"
} catch {
  Err 'Python not found on PATH. Install Python 3.10+ and reopen PowerShell.'
  exit 2
}

# venv
$venvPath = Join-Path -Path $projectRoot -ChildPath '.venv'
if (-not (Test-Path $venvPath)) {
  Info 'Creating virtual environment at .venv'
  & python -m venv .venv
} else {
  Info 'Using existing virtual environment at .venv'
}
$activateScript = Join-Path -Path $venvPath -ChildPath 'Scripts\Activate.ps1'
if (-not (Test-Path $activateScript)) { Err "Missing $activateScript"; exit 2 }
Info 'Activating virtual environment'
. $activateScript

# pip + deps
Info 'Upgrading pip'
python -m pip install --upgrade pip
if (-not (Test-Path 'requirements.txt')) { Err 'requirements.txt not found'; exit 2 }
Info 'Installing dependencies (this may take a few minutes)'
pip install -r requirements.txt

# Smoke test
$configPath = 'configs\model.sac_lstm.small.yaml'
if (-not (Test-Path $configPath)) { Warn "Config not found at $configPath - using defaults." }
Info 'Launching smoke test (SAC baseline)'
python -m src.saclstm_au.training.train_sac_lstm --config $configPath --run smoke

Write-Host ''
Info 'Smoke test complete'
Write-Host '  - Check experiments\smoke\metrics.csv'
Write-Host '  - Check artifacts\models\sac_lstm\*.zip'
