[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$upstreamsPath = Join-Path $projectRoot "config\upstreams.json"
$promptsPath = Join-Path $projectRoot "config\prompts.json"

$upstreams = Get-Content -Raw -LiteralPath $upstreamsPath | ConvertFrom-Json
$prompts = Get-Content -Raw -LiteralPath $promptsPath | ConvertFrom-Json

if ($upstreams.schema_version -ne 1) { throw "Unexpected upstream pin schema version." }
if ($upstreams.bonsai_demo.commit -notmatch '^[0-9a-f]{40}$') { throw "Bonsai demo commit must be a full SHA." }
if ($upstreams.llama_cpp_prism.commit -notmatch '^[0-9a-f]{40}$') { throw "llama.cpp commit must be a full SHA." }
if ($upstreams.model.revision -notmatch '^[0-9a-f]{40}$') { throw "Model revision must be a full SHA." }
if (@($prompts.prompts).Count -lt 5) { throw "The baseline corpus must contain at least five prompts." }

$duplicateIds = @($prompts.prompts | Group-Object id | Where-Object Count -gt 1)
if ($duplicateIds.Count -gt 0) { throw "Prompt IDs must be unique." }

foreach ($prompt in $prompts.prompts) {
    if (-not $prompt.id -or -not $prompt.category -or -not $prompt.prompt) {
        throw "Every prompt needs id, category, and prompt fields."
    }
    if ($prompt.max_tokens -lt 1) { throw "Prompt '$($prompt.id)' has invalid max_tokens." }
}

Write-Host "Configuration validation passed: $(@($prompts.prompts).Count) prompts and three pinned revisions."
