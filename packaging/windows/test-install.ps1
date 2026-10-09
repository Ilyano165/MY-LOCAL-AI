<#
  Echter Installationstest (CI auf windows-latest, oder manuell auf einer Test-VM):

    pwsh packaging\windows\test-install.ps1 -OldMsi dist\NOVA-0.1.0-x64.msi `
         -NewMsi dist\upgrade\NOVA-0.1.1-x64.msi -OldVersion 0.1.0 -NewVersion 0.1.1 -LogDir logs

  Prüft: Installation, Startmenü, Version (exe/ARP/Health), Hintergrunddienst, Benutzerdaten
  anlegen, Upgrade (Daten bleiben, Dienst wird vorher gestoppt), Downgrade-Sperre, Reparatur
  (gelöschte Programmdatei kommt zurück), Deinstallation (Programmordner weg, Daten bleiben).
  Schreibt logs\install-test.json + .md. Exit-Code ≠ 0 bei jedem Fehlschlag.
#>
param(
  [Parameter(Mandatory)] [string]$OldMsi,
  [Parameter(Mandatory)] [string]$NewMsi,
  [Parameter(Mandatory)] [string]$OldVersion,
  [Parameter(Mandatory)] [string]$NewVersion,
  [string]$LogDir = "logs",
  [int]$Port = 8765
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogDir = (Resolve-Path $LogDir).Path
$OldMsi = (Resolve-Path $OldMsi).Path
$NewMsi = (Resolve-Path $NewMsi).Path

$programDir = Join-Path $env:LOCALAPPDATA "Programs\NOVA"
$dataDir = Join-Path $env:LOCALAPPDATA "NOVA"
$nova = Join-Path $programDir "nova.exe"
$menu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\NOVA"
$results = [System.Collections.Generic.List[object]]::new()
$failed = $false

function Check([string]$name, [scriptblock]$test) {
  try {
    $detail = & $test
    $results.Add([pscustomobject]@{ check = $name; ok = $true; detail = "$detail" })
    Write-Host "PASS  $name  $detail"
  } catch {
    $script:failed = $true
    $results.Add([pscustomobject]@{ check = $name; ok = $false; detail = "$($_.Exception.Message)" })
    Write-Host "FAIL  $name  $($_.Exception.Message)"
  }
}

function Msi([string[]]$arguments, [string]$log) {
  $p = Start-Process msiexec.exe -ArgumentList ($arguments + @("/qn", "/norestart", "/l*v", "`"$LogDir\$log`"")) -Wait -PassThru
  return $p.ExitCode
}

function Health() {
  try { return Invoke-RestMethod "http://127.0.0.1:$Port/api/v1/health" -TimeoutSec 3 } catch { return $null }
}

# Registrierte Produkte über die Windows-Installer-API – dieselbe Quelle wie „Apps & Features“
# (Pro-Benutzer-MSIs stehen nicht unter HKCU\...\Uninstall, sondern in der Installer-Datenbank).
$UpgradeCode = "{872F4C49-83EC-40A2-8432-D306FA693A1B}"
function InstalledProducts() {
  $installer = New-Object -ComObject WindowsInstaller.Installer
  $found = @()
  foreach ($code in $installer.RelatedProducts($UpgradeCode)) {
    $found += [pscustomobject]@{
      Code = $code
      Name = $installer.ProductInfo($code, "ProductName")
      Version = $installer.ProductInfo($code, "VersionString")
    }
  }
  return ,$found
}

function ArpVersion() {
  $products = InstalledProducts
  if ($products.Count -eq 0) { return $null }
  return ($products | ForEach-Object Version) -join ","
}

function Expect([bool]$condition, [string]$message) { if (-not $condition) { throw $message } }

# ---------------------------------------------------------------- Installation
Check "install $OldVersion (exit 0)" {
  $code = Msi @("/i", "`"$OldMsi`"") "install.log"
  Expect ($code -eq 0) "msiexec exit code $code (see install.log)"
  "exit 0"
}
Check "program files in %LOCALAPPDATA%\Programs\NOVA" {
  Expect (Test-Path $nova) "nova.exe missing"
  Expect (Test-Path (Join-Path $programDir "nova-launcher.exe")) "nova-launcher.exe missing"
  Expect (Test-Path (Join-Path $programDir "_internal\api\static\index.html")) "UI files missing"
  $programDir
}
Check "Start Menu shortcuts" {
  Expect (Test-Path (Join-Path $menu "NOVA.lnk")) "NOVA.lnk missing"
  Expect (Test-Path (Join-Path $menu "Stop NOVA service.lnk")) "stop shortcut missing"
  "NOVA.lnk, Stop NOVA service.lnk"
}
Check "desktop shortcut not created by default" {
  Expect (-not (Test-Path (Join-Path ([Environment]::GetFolderPath("Desktop")) "NOVA.lnk"))) "unexpected desktop shortcut"
  "absent"
}
Check "version $OldVersion (exe + Windows Installer registration)" {
  $v = (& $nova --version).Trim()
  Expect ($v -eq "NOVA $OldVersion") "nova --version = '$v'"
  $arp = ArpVersion
  Expect ($arp -eq $OldVersion) "ARP DisplayVersion = '$arp'"
  "$v / ARP $arp"
}
Check "background service starts (setup mode, no model)" {
  & $nova service start --port $Port | Out-File "$LogDir\service-start-1.json"
  $h = Health
  Expect ($null -ne $h) "health endpoint not reachable"
  Expect ($h.version -eq $OldVersion) "health version $($h.version)"
  $s = Invoke-RestMethod "http://127.0.0.1:$Port/system/status" -Headers @{ "X-NOVA-Client" = "ci" }
  Expect ($s.mode -eq "setup") "mode = $($s.mode)"
  "health ok, mode=$($s.mode)"
}
Check "service listens on loopback only" {
  $listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop
  foreach ($l in $listeners) { Expect ($l.LocalAddress -in @("127.0.0.1", "::1")) "listening on $($l.LocalAddress)" }
  ($listeners | ForEach-Object LocalAddress) -join ","
}
Check "create user data (integration key + marker file)" {
  & $nova integrations create --name ci-check --scope models:read | Out-File "$LogDir\integration-create.json"
  Expect ($LASTEXITCODE -eq 0) "integrations create failed"
  Set-Content (Join-Path $dataDir "ci-marker.txt") "keep me"
  Expect (Test-Path (Join-Path $dataDir "integrations.db")) "integrations.db missing"
  $dataDir
}

# ---------------------------------------------------------------- Upgrade (Dienst läuft!)
Check "upgrade to $NewVersion while service is running (exit 0, no reboot)" {
  $code = Msi @("/i", "`"$NewMsi`"") "upgrade.log"
  Expect ($code -eq 0) "msiexec exit code $code (3010 = files were locked; see upgrade.log)"
  "exit 0"
}
Check "version after upgrade" {
  $v = (& $nova --version).Trim()
  Expect ($v -eq "NOVA $NewVersion") "nova --version = '$v'"
  Expect ((ArpVersion) -eq $NewVersion) "ARP shows $(ArpVersion)"
  Expect ((InstalledProducts).Count -eq 1) "more than one NOVA product registered"
  $v
}
Check "user data kept after upgrade" {
  Expect ((Get-Content (Join-Path $dataDir "ci-marker.txt")) -eq "keep me") "marker lost"
  $list = & $nova integrations list | Out-String
  Expect ($list -match "ci-check") "integration lost"
  "marker + integration present"
}
Check "service runs new version after restart" {
  & $nova service start --port $Port | Out-File "$LogDir\service-start-2.json"
  $h = Health
  Expect ($null -ne $h -and $h.version -eq $NewVersion) "health: $($h | ConvertTo-Json -Compress)"
  "health version $($h.version)"
}

# ---------------------------------------------------------------- Downgrade-Sperre
Check "downgrade to $OldVersion is refused" {
  $code = Msi @("/i", "`"$OldMsi`"") "downgrade.log"
  Expect ($code -ne 0) "downgrade was accepted"
  Expect ((& $nova --version).Trim() -eq "NOVA $NewVersion") "installed version changed"
  "refused with exit $code"
}

# ---------------------------------------------------------------- Reparatur
Check "repair restores a deleted program file" {
  $victim = Join-Path $programDir "_internal\api\static\index.html"
  Remove-Item $victim
  $code = Msi @("/fa", "`"$NewMsi`"") "repair.log"
  Expect ($code -eq 0) "msiexec exit code $code"
  Expect (Test-Path $victim) "file not restored"
  Expect ((Get-Content (Join-Path $dataDir "ci-marker.txt")) -eq "keep me") "repair touched user data"
  "index.html restored, data intact"
}

# ---------------------------------------------------------------- Deinstallation (Dienst läuft!)
Check "uninstall while service is running (exit 0)" {
  & $nova service start --port $Port | Out-Null
  $code = Msi @("/x", "`"$NewMsi`"") "uninstall.log"
  Expect ($code -eq 0) "msiexec exit code $code"
  "exit 0"
}
Check "program files and shortcuts removed" {
  Expect (-not (Test-Path $nova)) "nova.exe still present"
  Expect (-not (Test-Path $menu)) "Start Menu folder still present"
  Expect ($null -eq (ArpVersion)) "still listed in Add/Remove Programs"
  $left = if (Test-Path $programDir) { (Get-ChildItem -Recurse $programDir | Measure-Object).Count } else { 0 }
  Expect ($left -eq 0) "$left items left in $programDir"
  "removed"
}
Check "service stopped by uninstall" {
  Start-Sleep -Seconds 2
  Expect ($null -eq (Health)) "service still answering"
  "stopped"
}
Check "user data kept after uninstall" {
  Expect ((Get-Content (Join-Path $dataDir "ci-marker.txt")) -eq "keep me") "marker lost"
  Expect (Test-Path (Join-Path $dataDir "integrations.db")) "integrations.db lost"
  $dataDir
}

# ---------------------------------------------------------------- Bericht
$results | ConvertTo-Json -Depth 3 | Out-File "$LogDir\install-test.json" -Encoding utf8
$md = @("# NOVA installer test", "", "Old: $OldVersion · New: $NewVersion · $(Get-Date -Format o) · $([Environment]::OSVersion.VersionString)", "", "| Check | Result | Detail |", "|---|---|---|")
foreach ($r in $results) { $md += "| $($r.check) | $(if ($r.ok) { 'PASS' } else { 'FAIL' }) | $($r.detail -replace '\|', '/') |" }
$md | Out-File "$LogDir\install-test.md" -Encoding utf8
if ($failed) { Write-Host "Installer test FAILED"; exit 1 }
Write-Host "Installer test passed ($($results.Count) checks)"
