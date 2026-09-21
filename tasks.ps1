# PowerShell mirror of the Makefile for machines without GNU make. Same target names.
#   .\tasks.ps1 test | lint | fmt | replay <session> | scenarios | run | bench | venv | lock | install | install-dev | precommit
param(
    [Parameter(Position = 0, Mandatory = $true)] [string] $Target,
    [Parameter(Position = 1)] [string] $Arg
)
$ErrorActionPreference = "Stop"
$PY = if ($env:ALLY_PY) { $env:ALLY_PY } else { ".venv\Scripts\python.exe" }
$TORCH_INDEX = if ($env:TORCH_INDEX) { $env:TORCH_INDEX } else { "https://download.pytorch.org/whl/cu126" }

function Run {
    param([string[]] $cmd)
    Write-Host "> $($cmd -join ' ')"
    & $cmd[0] @($cmd | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
function ReqFile { param($stem) if (Test-Path "$stem.txt") { "$stem.txt" } else { "$stem.in" } }

switch ($Target) {
    "venv" {
        Run @("uv", "python", "install", "3.11")
        Run @("uv", "venv", "--python", "3.11", ".venv")
    }
    "lock" {
        Run @("uv", "pip", "compile", "requirements-dev.in", "-o", "requirements-dev.txt", "--python-version", "3.11", "--universal")
        Run @("uv", "pip", "compile", "requirements.in", "-o", "requirements.txt", "--python-version", "3.11", "--universal",
              "--extra-index-url", $TORCH_INDEX, "--index-strategy", "unsafe-best-match")
    }
    "install-dev" {
        Run @("uv", "pip", "install", "--python", $PY, "-r", (ReqFile "requirements-dev"))
        Run @("uv", "pip", "install", "--python", $PY, "-e", ".", "--no-deps")
    }
    "install" {
        & $PSCommandPath install-dev
        Run @("uv", "pip", "install", "--python", $PY, "-r", (ReqFile "requirements"),
              "--extra-index-url", $TORCH_INDEX, "--index-strategy", "unsafe-best-match")
    }
    "precommit" {
        Run @($PY, "-m", "pre_commit", "install")
        Run @($PY, "-m", "pre_commit", "run", "--all-files")
    }
    "test"       { Run @($PY, "-m", "pytest", "tests/unit", "tests/replay") }
    "lint"       { Run @($PY, "-m", "ruff", "check", "."); Run @($PY, "-m", "ruff", "format", "--check", ".") }
    "fmt"        { Run @($PY, "-m", "ruff", "check", "--fix", "."); Run @($PY, "-m", "ruff", "format", ".") }
    "replay"     { if (-not $Arg) { throw "usage: .\tasks.ps1 replay <session>" }; Run @($PY, "-m", "ally.main", "--replay", $Arg) }
    "scenarios"  { Run @($PY, "scripts/run_scenarios.py", "--repeat", "1") }
    "run"        { Run @($PY, "-m", "ally.main") }
    "bench"      { Run @($PY, "scripts/benchmark_fps.py") }
    "train-fall" { Run @($PY, "scripts/train_fall.py") }
    "eval-fall"  { Run @($PY, "scripts/eval_fall.py") }
    "train-fer"  { Run @($PY, "scripts/train_fer.py") }
    "eval-fer"   { Run @($PY, "scripts/eval_fer.py") }
    default      { throw "unknown target '$Target' - see Makefile for the list" }
}
