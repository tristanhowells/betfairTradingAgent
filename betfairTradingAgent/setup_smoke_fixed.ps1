
<# 
  setup_smoke.ps1 — Windows PowerShell
  Creates venv, installs deps, verifies imports (PowerShell-safe), runs smoke test.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err($m){ Write-Host "[ERROR] $m" -ForegroundColor Red }

# Locate project root (script in repo root OR in parent of sac_lstm_au_racing)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir
if (-not (Test-Path (Join-Path $projectRoot "requirements.txt"))) {
  if (Test-Path (Join-Path $projectRoot "sac_lstm_au_racing" "requirements.txt")) {
    $projectRoot = Join-Path $projectRoot "sac_lstm_au_racing"
  } else {
    Write-Err "requirements.txt not found in '$projectRoot' or '$projectRoot\sac_lstm_au_racing'."
    exit 2
  }
}
Write-Info "Project root: $projectRoot"
Set-Location $projectRoot

# Ensure Python
try {
  $pyVersion = & python -c "import sys; print(sys.version.split()[0])"
  Write-Info "Python detected: $pyVersion"
} catch {
  Write-Err "Python not found on PATH. Install Python 3.10+ and reopen PowerShell."
  exit 2
}

# venv
$venvPath = Join-Path $projectRoot ".venv"
if (-not (Test-Path $venvPath)) {
  Write-Info "Creating virtual environment at .venv"
  & python -m venv .venv
} else {
  Write-Info "Using existing virtual environment at .venv"
}
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) { Write-Err "Missing $activateScript"; exit 2 }
Write-Info "Activating virtual environment"
. $activateScript

# pip + deps
Write-Info "Upgrading pip"
python -m pip install --upgrade pip
if (-not (Test-Path "requirements.txt")) { Write-Err "requirements.txt not found"; exit 2 }
Write-Info "Installing dependencies (this may take a few minutes)"
pip install -r requirements.txt

# ----- Verification (PowerShell-safe): write a temp .py and run it -----
$verifyPy = @"
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
"@

$tmpPy = Join-Path $env:TEMP ("verify_imports_" + [guid]::NewGuid().ToString("N") + ".py")
$verifyPy | Set-Content -Path $tmpPy -Encoding UTF8
Write-Info "Verifying key packages via $tmpPy"
python $tmpPy
Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
# ----------------------------------------------------------------------

# Smoke test
$configPath = "configs\model.sac_lstm.small.yaml"
if (-not (Test-Path $configPath)) { Write-Warn "Config not found at $configPath — using defaults." }

Write-Info "Launching smoke test (SAC baseline)"
python -m src.saclstm_au.training.train_sac_lstm --config $configPath --run smoke

Write-Host ''
Write-Info 'Smoke test complete'
Write-Host '  - Check experiments\smoke\metrics.csv'
Write-Host '  - Check artifacts\models\sac_lstm\*.zip'
