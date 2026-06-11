# =========================================================
# Enterprise ML Project Tree Generator
# =========================================================

$MaxFileSizeMB = 100
$MaxDepth = 6

$excludeDirPatterns = @(
    '^\.git$',
    '^\.venv$',
    '^venv$',
    '^venv_',
    '^env$',
    '^env_',
    '^__pycache__$',
    '^\.pytest_cache$',
    '^\.mypy_cache$',
    '^\.ruff_cache$',
    '^node_modules$',
    '^dist$',
    '^build$',
    '^mlruns$',
    '^wandb$',
    '^checkpoints$',
    '^outputs$',
    '^artifacts$'
)

$excludeDirRegex = [regex]::new(
    '(' + ($excludeDirPatterns -join '|') + ')',
    [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
)

$excludeExtensions = @(
    '.pyc',
    '.pyo',
    '.log',
    '.tmp',
    '.sqlite3',
    '.db',
    '.pt',
    '.pth',
    '.onnx',
    '.h5',
    '.ckpt',
    '.bin',
    '.safetensors',
    '.npy',
    '.npz',
    '.dll',
    '.exe'
)

$excludeExtensions = $excludeExtensions |
ForEach-Object { $_.ToLowerInvariant() }

$maxFileSize = $MaxFileSizeMB * 1MB

function Format-FileSize {
    param([Int64]$Bytes)

    if ($Bytes -ge 1GB) {
        "{0:N2} GB" -f ($Bytes / 1GB)
    }
    elseif ($Bytes -ge 1MB) {
        "{0:N2} MB" -f ($Bytes / 1MB)
    }
    elseif ($Bytes -ge 1KB) {
        "{0:N2} KB" -f ($Bytes / 1KB)
    }
    else {
        "$Bytes B"
    }
}

function Show-Tree {
    param(
        [string]$Path,
        [string]$Prefix = "",
        [int]$Depth = 0
    )

    if ($Depth -ge $MaxDepth) {
        return
    }

    $items = Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue

    $filtered = [System.Collections.Generic.List[object]]::new()

    foreach ($item in $items) {

        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            continue
        }

        if ($item.PSIsContainer) {

            if ($excludeDirRegex.IsMatch($item.Name)) {
                continue
            }

            $filtered.Add($item)
        }
        else {

            $extension = ""

            if (-not [string]::IsNullOrEmpty($item.Extension)) {
                $extension = $item.Extension.ToLowerInvariant()
            }

            if ($excludeExtensions -contains $extension) {
                continue
            }

            if ($item.Length -ge $maxFileSize) {
                continue
            }

            $filtered.Add($item)
        }
    }

    $sorted = $filtered |
        Sort-Object `
            @{Expression="PSIsContainer";Descending=$true},
            @{Expression={ $_.Name.ToLowerInvariant() }}

    for ($i = 0; $i -lt $sorted.Count; $i++) {

        $item = $sorted[$i]

        $isLast = ($i -eq $sorted.Count - 1)

        $connector = if ($isLast) { "└── " } else { "├── " }

        if ($item.PSIsContainer) {

            "$Prefix$connector[D] $($item.Name)/"

            $childPrefix = if ($isLast) {
                "$Prefix    "
            }
            else {
                "$Prefix│   "
            }

            Show-Tree `
                -Path $item.FullName `
                -Prefix $childPrefix `
                -Depth ($Depth + 1)
        }
        else {

            $size = Format-FileSize $item.Length

            "$Prefix$connector[F] $($item.Name) ($size)"
        }
    }
}

$rootName = Split-Path (Get-Location).Path -Leaf

Set-Content `
    -Path "project_tree.txt" `
    -Value "$rootName/" `
    -Encoding UTF8

Show-Tree -Path (Get-Location).Path |
Add-Content `
    -Path "project_tree.txt" `
    -Encoding UTF8

Write-Host ""
Write-Host "Project tree generated successfully"
Write-Host "Output: project_tree.txt"