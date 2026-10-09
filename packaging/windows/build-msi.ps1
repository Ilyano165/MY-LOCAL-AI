<#
  Baut das MSI aus dist\NOVA (vorher: python packaging\build.py).

    pwsh packaging\windows\build-msi.ps1 [-Version 1.2.3] [-SourceDir dist\NOVA] [-OutDir dist]

  Voraussetzung: WiX Toolset v5.0.2 als .NET-Tool
    dotnet tool install --global wix --version 5.0.2
    wix extension add -g WixToolset.UI.wixext/5.0.2
#>
param(
  [string]$Version = "",
  [string]$SourceDir = "dist\NOVA",
  [string]$OutDir = "dist",
  [switch]$SkipValidation
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

if (-not $Version) {
  $Version = (Get-Content "$SourceDir\_internal\VERSION" -Raw).Trim()
}
# MSI-ProductVersion: nur Zahlen (major.minor.build) – Vorabversionen (z. B. 1.0.0rc1) sind
# im MSI nicht darstellbar und werden abgelehnt statt still umgedeutet.
if ($Version -notmatch '^\d{1,3}\.\d{1,3}\.\d{1,5}$') {
  throw "Version '$Version' is not a valid MSI version (expected X.Y.Z)"
}
$source = (Resolve-Path $SourceDir).Path
foreach ($exe in @("nova.exe", "nova-launcher.exe", "nova-desktop.exe")) {
  if (-not (Test-Path "$source\$exe")) { throw "$exe missing in $source – run packaging\build.py first" }
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$msi = Join-Path (Resolve-Path $OutDir).Path "NOVA-$Version-x64.msi"

& wix build packaging\windows\nova.wxs `
  -arch x64 `
  -ext WixToolset.UI.wixext `
  -d "Version=$Version" `
  -d "SourceDir=$source" `
  -d "NoticeRtf=$root\packaging\windows\notice.rtf" `
  -d "IconFile=$root\packaging\assets\nova.ico" `
  -o $msi
if ($LASTEXITCODE -ne 0) { throw "wix build failed with exit code $LASTEXITCODE" }
Write-Host "Built $msi"

# ICE-Validierung (Windows Installer Internal Consistency Evaluators). Unterdrückt sind nur
# Regeln, die für Pro-Benutzer-Installationen in %LOCALAPPDATA% nicht passen:
#   ICE38/ICE64 – Komponenten im Benutzerprofil sollen HKCU-Schlüsselpfade/RemoveFolder haben;
#                 die automatisch eingesammelten Dateien (Files) haben Datei-Schlüsselpfade
#   ICE91       – Warnung zu Dateien in Profilordnern bei Pro-Benutzer-Paketen
if (-not $SkipValidation) {
  & wix msi validate -sice ICE38 -sice ICE64 -sice ICE91 $msi
  if ($LASTEXITCODE -ne 0) { throw "MSI validation (ICE) failed with exit code $LASTEXITCODE" }
  Write-Host "ICE validation passed"
}
