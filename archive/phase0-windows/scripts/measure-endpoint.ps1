[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$PromptId = "smoke_exact_001",
    [ValidateRange(1, 20)][int]$WarmupRuns = 1,
    [ValidateRange(1, 30)][int]$MeasuredRuns = 5,
    [ValidateRange(10, 3600)][int]$TimeoutSeconds = 300,
    [string]$EnvironmentManifestPath,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\artifacts\endpoint")
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$promptPath = Join-Path $projectRoot "config\prompts.json"
$upstreamPath = Join-Path $projectRoot "config\upstreams.json"
$corpus = Get-Content -Raw -LiteralPath $promptPath | ConvertFrom-Json
$prompt = @($corpus.prompts | Where-Object { $_.id -eq $PromptId })
if ($prompt.Count -ne 1) { throw "Expected one prompt with ID '$PromptId'; found $($prompt.Count)." }
$prompt = $prompt[0]

if (-not $EnvironmentManifestPath) {
    $candidate = Get-ChildItem -LiteralPath (Join-Path $projectRoot "artifacts\environment") -Filter "machine-state-*.json" -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $candidate) { throw "Capture a machine-state manifest before measuring." }
    $EnvironmentManifestPath = $candidate.FullName
}
$EnvironmentManifestPath = (Resolve-Path -LiteralPath $EnvironmentManifestPath).Path

$health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
if ($health.status -ne "ok") { throw "Server is not healthy at $BaseUrl." }

$modelList = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -TimeoutSec 5
$modelId = @($modelList.data)[0].id
if (-not $modelId) { throw "Server did not expose a model ID." }

$body = [ordered]@{
    model = $modelId
    messages = @(@{ role = "user"; content = $prompt.prompt })
    temperature = 0
    max_tokens = [int]$prompt.max_tokens
    stream = $false
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 10 -Compress

$samples = @()
for ($iteration = 1; $iteration -le ($WarmupRuns + $MeasuredRuns); $iteration++) {
    $kind = if ($iteration -le $WarmupRuns) { "warmup" } else { "measured" }
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec $TimeoutSeconds
    $timer.Stop()

    $choice = @($response.choices)[0]
    $content = [string]$choice.message.content
    $expectedExactPass = if ($null -ne $prompt.expected_exact) { $content.Trim() -ceq [string]$prompt.expected_exact } else { $null }
    $expectedJsonPass = $null
    if ($null -ne $prompt.expected_json) {
        try {
            $actualJson = $content | ConvertFrom-Json
            $expectedJsonPass = $true
            foreach ($property in $prompt.expected_json.PSObject.Properties) {
                $actualValue = $actualJson.($property.Name)
                if ($actualValue -ne $property.Value) { $expectedJsonPass = $false }
            }
        } catch {
            $expectedJsonPass = $false
        }
    }

    $sample = [ordered]@{
        kind = $kind
        iteration = $iteration
        captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        wall_seconds = [math]::Round($timer.Elapsed.TotalSeconds, 4)
        finish_reason = $choice.finish_reason
        content = $content
        expected_exact_pass = $expectedExactPass
        expected_json_pass = $expectedJsonPass
        prompt_tokens = $response.timings.prompt_n
        prompt_ms = $response.timings.prompt_ms
        prompt_tokens_per_second = $response.timings.prompt_per_second
        generated_tokens = $response.timings.predicted_n
        generated_ms = $response.timings.predicted_ms
        generated_tokens_per_second = $response.timings.predicted_per_second
        cached_prompt_tokens = $response.timings.cache_n
    }
    $samples += $sample
    Write-Host "$kind $iteration`: wall=$($sample.wall_seconds)s prompt=$($sample.prompt_tokens_per_second) tok/s decode=$($sample.generated_tokens_per_second) tok/s finish=$($sample.finish_reason)"
}

$measured = @($samples | Where-Object kind -eq "measured")
$decodeRates = @($measured | ForEach-Object { [double]$_.generated_tokens_per_second } | Sort-Object)
$wallTimes = @($measured | ForEach-Object { [double]$_.wall_seconds } | Sort-Object)
$middle = [int][math]::Floor($MeasuredRuns / 2)
$medianDecode = if ($MeasuredRuns % 2) { $decodeRates[$middle] } else { ($decodeRates[$middle - 1] + $decodeRates[$middle]) / 2 }
$medianWall = if ($MeasuredRuns % 2) { $wallTimes[$middle] } else { ($wallTimes[$middle - 1] + $wallTimes[$middle]) / 2 }
$absoluteDeviations = @($decodeRates | ForEach-Object { [math]::Abs($_ - $medianDecode) } | Sort-Object)
$decodeMad = if ($MeasuredRuns % 2) { $absoluteDeviations[$middle] } else { ($absoluteDeviations[$middle - 1] + $absoluteDeviations[$middle]) / 2 }

$result = [ordered]@{
    schema_version = 1
    captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    label = "performance-measured-only-if-correctness-and-environment-are-reviewed"
    endpoint = $BaseUrl
    model_id = $modelId
    prompt_id = $PromptId
    prompt_corpus_sha256 = (Get-FileHash -LiteralPath $promptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    upstream_pins_sha256 = (Get-FileHash -LiteralPath $upstreamPath -Algorithm SHA256).Hash.ToLowerInvariant()
    environment_manifest = $EnvironmentManifestPath
    environment_manifest_sha256 = (Get-FileHash -LiteralPath $EnvironmentManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    request = $body | ConvertFrom-Json
    protocol = [ordered]@{ warmup_runs = $WarmupRuns; measured_runs = $MeasuredRuns }
    summary = [ordered]@{
        median_decode_tokens_per_second = $medianDecode
        median_decode_absolute_deviation = $decodeMad
        median_wall_seconds = $medianWall
        all_exact_checks_passed = if ($null -ne $prompt.expected_exact) { @($measured | Where-Object expected_exact_pass -eq $false).Count -eq 0 } else { $null }
        all_json_checks_passed = if ($null -ne $prompt.expected_json) { @($measured | Where-Object expected_json_pass -eq $false).Count -eq 0 } else { $null }
    }
    samples = $samples
}

$resolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutputDirectory -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultPath = Join-Path $resolvedOutputDirectory "$PromptId-$timestamp.json"
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resultPath -Encoding UTF8
Write-Host "Result: $resultPath"
Write-Host "Median decode: $([math]::Round($medianDecode, 3)) tok/s (MAD $([math]::Round($decodeMad, 3)))"
