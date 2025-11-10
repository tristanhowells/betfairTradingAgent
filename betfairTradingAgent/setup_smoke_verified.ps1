
# setup_smoke_verified.ps1 - setup + import verification (ASCII-only)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Err ($m){ Write-Host "[ERROR] $m" -ForegroundColor Red }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir
if (-not (Test-Path (Join-Path $projectRoot 'requirements.txt'))) {
  if (Test-Path (Join-Path $projectRoot 'sac_lstm_au_racing' 'requirements.txt')) {
    $projectRoot = Join-Path $projectRoot 'sac_lstm_au_racing'
  } else {
    Err "requirements.txt not found in '$projectRoot' or '$projectRoot\sac_lstm_au_racing'."
    exit 2
  }
}
Info "Project root: $projectRoot"
Set-Location $projectRoot

try {
  $py = & python -c "import sys; print(sys.version.split()[0])"
  Info "Python detected: $py"
} catch {
  Err 'Python not found on PATH. Install Python 3.10+ and reopen PowerShell.'
  exit 2
}

$venvPath = Join-Path $projectRoot '.venv'
if (-not (Test-Path $venvPath)) {
  Info 'Creating virtual environment at .venv'
  & python -m venv .venv
} else {
  Info 'Using existing virtual environment at .venv'
}
$activateScript = Join-Path $venvPath 'Scripts\Activate.ps1'
if (-not (Test-Path $activateScript)) { Err "Missing $activateScript"; exit 2 }
Info 'Activating virtual environment'
. $activateScript

Info 'Upgrading pip'
python -m pip install --upgrade pip
if (-not (Test-Path 'requirements.txt')) { Err 'requirements.txt not found'; exit 2 }
Info 'Installing dependencies (this may take a few minutes)'
pip install -r requirements.txt

# Verification using single-quoted here-string (no interpolation). Keep markers at column 1.
$verifyPy = @'
import importlib
mods = ['gymnasium','stable_baselines3','numpy','torch']
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print('[WARN] Some packages failed to import:')
    for m, e in missing:
        print('   -', m, ':', e)
else:
    print('[INFO] All key packages imported successfully.')
'@
$tmpPy = Join-Path $env:TEMP ('verify_imports_' + [guid]::NewGuid().ToString('N') + '.py')
$verifyPy | Set-Content -Path $tmpPy -Encoding ASCII
Info "Verifying key packages via $tmpPy"
python $tmpPy
Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue

$configPath = 'configs\model.sac_lstm.small.yaml'
if (-not (Test-Path $configPath)) { Warn "Config not found at $configPath - using defaults." }
Info 'Launching smoke test (SAC baseline)'
python -m src.saclstm_au.training.train_sac_lstm --config $configPath --run smoke

Write-Host ''
Info 'Smoke test complete'
Write-Host '  - Check experiments\smoke\metrics.csv'
Write-Host '  - Check artifacts\models\sac_lstm\*.zip'
