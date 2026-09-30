param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$ServerName = "127.0.0.1",
  [int]$ServerPort = 7860
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

& $Python final\code\src_neurosymbolic\demo_ui\app.py `
  --server-name $ServerName `
  --server-port $ServerPort
