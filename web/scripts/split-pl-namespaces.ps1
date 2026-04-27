# ============================================
# Split pl-PL.json into namespace files
# Run from: web/ directory
# ============================================

$mergedFile = "src\i18n\pl-PL.json"
$outDir = "src\i18n\pl-PL"

if (-not (Test-Path $mergedFile)) {
    Write-Error "❌ Merged file not found: $mergedFile"
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Read JSON
$json = Get-Content $mergedFile -Raw -Encoding UTF8 | ConvertFrom-Json

# Each top-level key becomes a namespace file
foreach ($prop in $json.PSObject.Properties) {
    $name = $prop.Name
    $value = $prop.Value

    $outFile = Join-Path $outDir "$name.json"

    # Convert to JSON with proper indentation
    $value | ConvertTo-Json -Depth 100 | Out-File $outFile -Encoding UTF8

    Write-Host "✅ Created: $outFile"
}

Write-Host ""
Write-Host "🎉 All namespace files created in: $outDir"