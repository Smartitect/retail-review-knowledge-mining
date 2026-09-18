#! /usr/bin/pwsh
#
# Runs once after the dev container is created.
# Idempotent: safe to re-run by hand after a partial failure.

$ErrorActionPreference = "Stop"
# Without this, a *native* command (uv, apt, gh) returning non-zero does NOT stop
# the script. $ErrorActionPreference alone only governs PowerShell cmdlets.
$PSNativeCommandUseErrorActionPreference = $true

$workspace = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $workspace ".venv"

if ($env:CI -or $env:TF_BUILD) {
    # CI agents provision their own runtime and must not have one built underneath them.
    Write-Host "CI detected - skipping local environment setup."
}
else {
    Write-Host "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.sh | bash

    # The installer cannot modify the already-running shell, so extend PATH here.
    $uvBin = Join-Path $HOME ".local/bin"
    if ($env:PATH -notlike "*$uvBin*") { $env:PATH = "${uvBin}:$env:PATH" }

    Write-Host "Syncing the Python environment..."
    Push-Location $workspace
    try {
        uv sync --python 3.12
    }
    finally {
        Pop-Location
    }
}

# A virtual environment that must be activated manually will eventually not be.
# profile.ps1 is the AllHosts profile, so it covers the integrated terminal and a
# bare `pwsh` alike.
$profileDir = Join-Path $HOME ".config/powershell"
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null

@"
# Managed by .devcontainer/postCreateCommand.ps1 - edits will be overwritten.
`$uvBin = Join-Path `$HOME ".local/bin"
if ((Test-Path `$uvBin) -and (`$env:PATH -notlike "*`$uvBin*")) { `$env:PATH = "`${uvBin}:`$env:PATH" }

# Guarded: the environment does not exist on a run where uv sync was skipped.
`$venvActivate = "$venv/bin/activate.ps1"
if (Test-Path `$venvActivate) { . `$venvActivate }
"@ | Set-Content -Path (Join-Path $profileDir "profile.ps1") -Encoding utf8

Write-Host "Post-create complete."
