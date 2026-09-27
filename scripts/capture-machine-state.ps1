[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\artifacts\environment")
)

$ErrorActionPreference = "Stop"

function Invoke-CapturedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )

    $command = Get-Command $Executable -ErrorAction SilentlyContinue
    if (-not $command) {
        return [ordered]@{ available = $false; output = $null; exit_code = $null }
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $command.Source @Arguments 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    return [ordered]@{
        available = $true
        path = $command.Source
        output = $output.Trim()
        exit_code = $exitCode
    }
}

function Get-GitSnapshot {
    param([Parameter(Mandatory = $true)][string]$RepositoryPath)

    if (-not (Test-Path -LiteralPath (Join-Path $RepositoryPath ".git"))) {
        return [ordered]@{ available = $false; path = $RepositoryPath }
    }

    $commit = Invoke-CapturedCommand -Executable "git" -Arguments @("-C", $RepositoryPath, "rev-parse", "HEAD")
    $branch = Invoke-CapturedCommand -Executable "git" -Arguments @("-C", $RepositoryPath, "branch", "--show-current")
    $status = Invoke-CapturedCommand -Executable "git" -Arguments @("-C", $RepositoryPath, "status", "--porcelain=v1")
    $statusLines = @($status.output -split "`r?`n" | Where-Object { $_ })

    return [ordered]@{
        available = $true
        path = (Resolve-Path -LiteralPath $RepositoryPath).Path
        commit = $commit.output
        branch = $branch.output
        clean = ($statusLines.Count -eq 0)
        changed_path_count = $statusLines.Count
    }
}

function Get-ToolVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Arguments = @("--version")
    )

    $result = Invoke-CapturedCommand -Executable $Name -Arguments $Arguments
    if (-not $result.available) {
        return [ordered]@{ available = $false }
    }

    return [ordered]@{
        available = $true
        path = $result.path
        version = ($result.output -split "`r?`n" | Select-Object -First 1)
        exit_code = $result.exit_code
    }
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "..")).Path
$bonsaiDemoPath = Join-Path $workspaceRoot "work\sources\Bonsai-demo"
$llamaCppPath = Join-Path $workspaceRoot "work\sources\llama.cpp-prism"

$operatingSystem = Get-CimInstance Win32_OperatingSystem
$computerSystem = Get-CimInstance Win32_ComputerSystem
$processors = @(Get-CimInstance Win32_Processor)
$memoryModules = @(Get-CimInstance Win32_PhysicalMemory)
$bios = Get-CimInstance Win32_BIOS
$logicalDisk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$battery = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue)
$powerPlan = Invoke-CapturedCommand -Executable "powercfg.exe" -Arguments @("/getactivescheme")

$gpuQuery = Invoke-CapturedCommand -Executable "nvidia-smi" -Arguments @(
    "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,pstate,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
    "--format=csv,noheader,nounits"
)

$gpuProcesses = Invoke-CapturedCommand -Executable "nvidia-smi" -Arguments @(
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits"
)

$topProcesses = @(
    Get-Process |
        Sort-Object WorkingSet64 -Descending |
        Select-Object -First 15 Name, Id,
            @{ Name = "working_set_bytes"; Expression = { $_.WorkingSet64 } },
            @{ Name = "cpu_seconds"; Expression = { $_.CPU } }
)

$manifest = [ordered]@{
    schema_version = 1
    captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    captured_at_local = (Get-Date).ToString("o")
    privacy = [ordered]@{
        serial_numbers_excluded = $true
        network_addresses_excluded = $true
        environment_variables_excluded = $true
        command_lines_excluded = $true
    }
    project = [ordered]@{
        root = $projectRoot
        upstream_pins_file = (Join-Path $projectRoot "config\upstreams.json")
        prompt_corpus_file = (Join-Path $projectRoot "config\prompts.json")
    }
    operating_system = [ordered]@{
        caption = $operatingSystem.Caption
        version = $operatingSystem.Version
        build_number = $operatingSystem.BuildNumber
        architecture = $operatingSystem.OSArchitecture
        last_boot_time = $operatingSystem.LastBootUpTime.ToUniversalTime().ToString("o")
    }
    machine = [ordered]@{
        manufacturer = $computerSystem.Manufacturer
        model = $computerSystem.Model
        total_physical_memory_bytes = [int64]$computerSystem.TotalPhysicalMemory
        free_physical_memory_bytes = [int64]$operatingSystem.FreePhysicalMemory * 1KB
        bios_version = @($bios.SMBIOSBIOSVersion)
        bios_release_date = if ($bios.ReleaseDate) { $bios.ReleaseDate.ToString("o") } else { $null }
        ac_or_battery = if ($battery.Count -eq 0) { "no_battery_reported" } else { @($battery | Select-Object BatteryStatus, EstimatedChargeRemaining) }
        active_power_plan = $powerPlan.output
    }
    cpu = @($processors | ForEach-Object {
        [ordered]@{
            name = $_.Name.Trim()
            physical_cores = $_.NumberOfCores
            logical_processors = $_.NumberOfLogicalProcessors
            max_clock_mhz = $_.MaxClockSpeed
        }
    })
    memory_modules = @($memoryModules | ForEach-Object {
        [ordered]@{
            capacity_bytes = [int64]$_.Capacity
            speed_mt_s = $_.Speed
            configured_speed_mt_s = $_.ConfiguredClockSpeed
            data_width_bits = $_.DataWidth
        }
    })
    storage = [ordered]@{
        drive = "C:"
        size_bytes = [int64]$logicalDisk.Size
        free_bytes = [int64]$logicalDisk.FreeSpace
    }
    gpu = [ordered]@{
        query = $gpuQuery
        compute_processes = $gpuProcesses
    }
    top_processes_by_working_set = $topProcesses
    tools = [ordered]@{
        powershell = [ordered]@{ version = $PSVersionTable.PSVersion.ToString(); edition = $PSVersionTable.PSEdition }
        git = Get-ToolVersion -Name "git"
        python = Get-ToolVersion -Name "python"
        python_launcher = Get-ToolVersion -Name "py" -Arguments @("--version")
        uv = Get-ToolVersion -Name "uv"
        cmake = Get-ToolVersion -Name "cmake"
        nvidia_smi = Get-ToolVersion -Name "nvidia-smi"
        nvcc = Get-ToolVersion -Name "nvcc"
        nsys = Get-ToolVersion -Name "nsys"
        ncu = Get-ToolVersion -Name "ncu"
    }
    repositories = [ordered]@{
        experiment = Get-GitSnapshot -RepositoryPath $projectRoot
        bonsai_demo = Get-GitSnapshot -RepositoryPath $bonsaiDemoPath
        llama_cpp_prism = Get-GitSnapshot -RepositoryPath $llamaCppPath
    }
}

$resolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutputDirectory -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$manifestPath = Join-Path $resolvedOutputDirectory "machine-state-$timestamp.json"
$hashPath = "$manifestPath.sha256"

$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$hash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([System.IO.Path]::GetFileName($manifestPath))" | Set-Content -LiteralPath $hashPath -Encoding ASCII

Write-Host "Machine-state manifest: $manifestPath"
Write-Host "SHA-256: $hash"
