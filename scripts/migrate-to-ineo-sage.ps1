# ============================================
# INEO Sage Migration - Cleanup & Build Script
# Run AFTER applying all file replacements
# ============================================

param(
    [switch]$DryRun = $false,
    [switch]$RemoveDocs = $false
)

$ErrorActionPreference = "Stop"
$webRoot = "C:\Users\MirekWilmer\Projects\ApeRAG\web"

if (-not (Test-Path $webRoot)) {
    Write-Error "❌ Web root not found: $webRoot"
    exit 1
}

Set-Location $webRoot

Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  INEO Sage Migration Script" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE — no files will be modified" -ForegroundColor Yellow
    Write-Host ""
}

# ----- Step 1: Verify Polish files exist -----
Write-Host "📋 Step 1/5: Verifying Polish files..." -ForegroundColor Green

$requiredFiles = @(
    "src\i18n\pl-PL.json",
    "src\i18n\pl-PL\common.json",
    "src\i18n\pl-PL\page_auth.json",
    "src\i18n\pl-PL\activity.json",
    "src\services\cookies.ts",
    "src\i18n\request.ts"
)

$missing = @()
foreach ($f in $requiredFiles) {
    if (-not (Test-Path $f)) {
        $missing += $f
    }
}

if ($missing.Count -gt 0) {
    Write-Host "❌ Missing required files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Apply file replacements from previous messages first!" -ForegroundColor Yellow
    exit 1
}
Write-Host "   ✅ All required files present" -ForegroundColor Gray

# ----- Step 2: Verify cookies.ts has pl-PL -----
Write-Host ""
Write-Host "📋 Step 2/5: Verifying cookies.ts has 'pl-PL'..." -ForegroundColor Green
$cookiesContent = Get-Content "src\services\cookies.ts" -Raw
if ($cookiesContent -notmatch "pl-PL") {
    Write-Host "❌ cookies.ts does not contain 'pl-PL'!" -ForegroundColor Red
    Write-Host "   Apply the updated cookies.ts from Message 3 follow-up" -ForegroundColor Yellow
    exit 1
}
if ($cookiesContent -match "zh-CN") {
    Write-Host "⚠️  WARNING: cookies.ts still contains 'zh-CN'" -ForegroundColor Yellow
    Write-Host "   Make sure you replaced the entire file" -ForegroundColor Yellow
}
Write-Host "   ✅ cookies.ts has Polish locale" -ForegroundColor Gray

# ----- Step 3: Cleanup Chinese files -----
Write-Host ""
Write-Host "📋 Step 3/5: Cleaning up Chinese i18n files..." -ForegroundColor Green

$toRemove = @(
    "src\i18n\zh-CN",
    "src\i18n\zh-CN.json",
    "public\logo_dark_zh_CN.png",
    "public\logo_light_zh_CN.png"
)

if ($RemoveDocs) {
    $toRemove += @("docs\zh-CN", "web\docs\zh-CN", "README-zh.md")
}

foreach ($item in $toRemove) {
    if (Test-Path $item) {
        if ($DryRun) {
            Write-Host "   [DRY] Would remove: $item" -ForegroundColor Yellow
        } else {
            Remove-Item -Recurse -Force $item
            Write-Host "   🗑️  Removed: $item" -ForegroundColor Gray
        }
    }
}

# ----- Step 4: Search for remaining ApeRAG references -----
Write-Host ""
Write-Host "📋 Step 4/5: Scanning for remaining 'ApeRAG' references..." -ForegroundColor Green

$apeRagRefs = Get-ChildItem -Recurse -File -Include *.ts, *.tsx, *.json `
    | Where-Object { $_.FullName -notmatch '\\node_modules\\|\\\.next\\|\\src\\api\\' } `
    | Select-String -Pattern "ApeRAG" -CaseSensitive

if ($apeRagRefs) {
    Write-Host "   ⚠️  Found $($apeRagRefs.Count) remaining 'ApeRAG' references:" -ForegroundColor Yellow
    $apeRagRefs | Select-Object -First 10 | ForEach-Object {
        $relPath = $_.Path.Replace("$webRoot\", "")
        Write-Host "      $relPath`:$($_.LineNumber)" -ForegroundColor Yellow
    }
    if ($apeRagRefs.Count -gt 10) {
        Write-Host "      ... and $($apeRagRefs.Count - 10) more" -ForegroundColor Yellow
    }
    Write-Host "   💡 Review these manually — some may be intentional (e.g., GitHub URL)" -ForegroundColor Gray
} else {
    Write-Host "   ✅ No remaining 'ApeRAG' references in code" -ForegroundColor Gray
}

# ----- Step 5: Run i18n sync (if not dry run) -----
Write-Host ""
Write-Host "📋 Step 5/5: i18n sync..." -ForegroundColor Green

if ($DryRun) {
    Write-Host "   [DRY] Would run: yarn i18n:sync" -ForegroundColor Yellow
} else {
    if (Test-Path "node_modules") {
        try {
            yarn i18n:sync
            Write-Host "   ✅ i18n sync complete" -ForegroundColor Gray
        } catch {
            Write-Host "   ⚠️  i18n:sync failed — run manually after 'yarn install'" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ⚠️  node_modules missing — skipping. Run 'yarn install' first" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✅ Migration script complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Cyan
Write-Host "   1. yarn install     (if not yet)" -ForegroundColor White
Write-Host "   2. yarn dev         (start dev server)" -ForegroundColor White
Write-Host "   3. Open http://localhost:3000" -ForegroundColor White
Write-Host "   4. Test language switcher (English ↔ Polski)" -ForegroundColor White
Write-Host "   5. Test signup form (no Chinese 'email' label!)" -ForegroundColor White
Write-Host ""