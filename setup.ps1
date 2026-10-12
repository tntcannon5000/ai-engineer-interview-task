# Native Windows launcher; shared setup logic lives in scripts/setup.py.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv and add it to PATH: https://docs.astral.sh/uv/' }
    if ($args -contains '--help' -or $args -contains '-h') {
        uv run python scripts/setup.py --help
    } else {
        uv sync --locked --quiet
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        uv run python scripts/setup.py @args
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
