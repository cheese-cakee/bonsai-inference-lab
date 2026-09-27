[CmdletBinding()]
param(
    [ValidateSet("PTQ1_0", "PQ2_0")][string]$Quant = "PTQ1_0",
    [ValidateRange(0, 99)][int]$GpuLayers = 32,
    [ValidateRange(512, 32768)][int]$Context = 4096,
    [ValidateRange(1, 16)][int]$Parallel = 1,
    [ValidateRange(1024, 65535)][int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "..")).Path
$demoRoot = Join-Path $workspaceRoot "work\sources\Bonsai-demo"
$binaryDirectory = Join-Path $demoRoot "bin\cuda"
$server = Join-Path $binaryDirectory "llama-server.exe"
$model = Join-Path $demoRoot "models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-$Quant.gguf"
$releaseMarker = Join-Path $binaryDirectory ".llama_release"

if (-not (Test-Path -LiteralPath $server)) { throw "Prism llama-server binary is missing. Run the vendor setup first." }
if (-not (Test-Path -LiteralPath $model)) { throw "Model is missing: $model" }
if (-not (Test-Path -LiteralPath $releaseMarker)) { throw "Runtime release marker is missing." }

$release = (Get-Content -LiteralPath $releaseMarker -Raw).Trim()
if ($release -ne "prism-b10709-9a9394a cuda-12.4") {
    throw "Unexpected runtime release: $release. Update the experiment pins before comparing results."
}

$serverPresent = $false
try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
    $serverPresent = $true
} catch {
    # The health endpoint did not answer, which is the expected pre-launch state.
}
if ($serverPresent) { throw "A server already responds on port $Port. Choose another port or stop it." }

$modelSize = (Get-Item -LiteralPath $model).Length
if ($modelSize -lt 5GB) { throw "The model file is unexpectedly small ($modelSize bytes). Check the download." }

$env:Path = "$binaryDirectory;$env:Path"
$arguments = @(
    "-m", $model,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-ngl", "$GpuLayers",
    "-c", "$Context",
    "-np", "$Parallel",
    "-fa", "on",
    "--jinja",
    "--chat-template-kwargs", '{"enable_thinking": false}'
)

Write-Host "Model: $model"
Write-Host "Runtime: $release"
Write-Host "Preset: quant=$Quant gpu_layers=$GpuLayers context=$Context parallel=$Parallel text_only=true"
Write-Host "Endpoint: http://127.0.0.1:$Port"

& $server @arguments
exit $LASTEXITCODE
