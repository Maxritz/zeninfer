param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [int]$MaxNewTokens = 200,
    [string]$Prompt = "Once upon a time"
)
$ErrorActionPreference = "Stop"
$inspect = Join-Path $BuildDirectory "Release/zen_model_inspect.exe"
$generate = Join-Path $BuildDirectory "Release/zen_generate_ids.exe"
if (!(Test-Path $inspect)) { $inspect = Join-Path $BuildDirectory "zen_model_inspect.exe" }
if (!(Test-Path $generate)) { $generate = Join-Path $BuildDirectory "zen_generate_ids.exe" }
if (!(Test-Path $inspect) -or !(Test-Path $generate)) { throw "ZenInfer executables not found under $BuildDirectory" }
if (!(Test-Path $ModelPath)) { throw "GGUF fixture not found: $ModelPath" }
Write-Host "=== GGUF inspection ==="
& $inspect $ModelPath
if ($LASTEXITCODE -ne 0) { throw "zen_model_inspect failed with exit code $LASTEXITCODE" }
Write-Host "=== Real model generation ==="
$output = & $generate $ModelPath --prompt-text $Prompt --max-new $MaxNewTokens 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "zen_generate_ids failed: $output" }
$output
$prefill = [regex]::Match($output, 'prefill_tokens_per_second=([0-9.]+)')
$decode = [regex]::Match($output, 'tokengen_tokens_per_second=([0-9.]+)')
$ids = [regex]::Match($output, '(?m)^generated_ids=(.*)$')
if (!$prefill.Success -or !$decode.Success -or !$ids.Success) { throw "Missing generation or throughput fields" }
$count = @($ids.Groups[1].Value.Split(',') | Where-Object { $_ -match '^\d+$' }).Count
if ($count -lt 1) { throw "No generated tokens" }
Write-Host ("WINDOWS_SMOKE PASS generated_tokens={0} prefill_tok_s={1} tokengen_tok_s={2}" -f $count,$prefill.Groups[1].Value,$decode.Groups[1].Value)
