# One-click evidence refresh: build -> flash -> golden -> stream -> report.
# Usage: powershell -ExecutionPolicy Bypass -File stm32_deployment\tools\verify_all.ps1 [-Port COM3]
param([string]$Port = "COM3")

$ErrorActionPreference = "Continue"
$Root = "D:\dsc-amcnet-cortex-m7-amc"
$Proj = "D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx"
$UV4  = "D:\APP\Keil5\UV4\UV4.exe"
$Ev   = Join-Path $Root "stm32_deployment\results\evidence"
New-Item -ItemType Directory -Force $Ev | Out-Null

Write-Host "== 1/5 build ==" -ForegroundColor Cyan
$r = Start-Process $UV4 -ArgumentList @("-b", $Proj, "-j0", "-o", (Join-Path $Ev "build.log")) -Wait -PassThru
Write-Host "build exit=$($r.ExitCode)"

Write-Host "== 2/5 flash + boot capture ==" -ForegroundColor Cyan
Get-Process UV4 -ErrorAction SilentlyContinue | Stop-Process -Force
$serial = New-Object System.IO.Ports.SerialPort $Port,115200,None,8,One
try { $serial.Open() } catch { Write-Host "COM port busy: $($_.Exception.Message)" -ForegroundColor Red; exit 1 }
$sb = New-Object System.Text.StringBuilder
$f = Start-Process $UV4 -ArgumentList @("-f", $Proj, "-o", (Join-Path $Ev "flash.log")) -Wait -PassThru
Write-Host "flash exit=$($f.ExitCode)"
$end = (Get-Date).AddSeconds(18)
while ((Get-Date) -lt $end) {
    try { $s = $serial.ReadExisting(); if ($s) { [void]$sb.Append($s) } } catch {}
    Start-Sleep -Milliseconds 100
}
$serial.Close()
$sb.ToString() | Set-Content (Join-Path $Ev "serial_boot.log") -Encoding UTF8
(Get-Content (Join-Path $Ev "serial_boot.log") | Select-String -Pattern "\[DWT\] N=1000") | ForEach-Object { Write-Host "  $($_.Line)" }

Write-Host "== 3/5 golden vectors ==" -ForegroundColor Cyan
python (Join-Path $Root "stm32_deployment\tools\golden_test.py") --port $Port --flash 2>&1 |
    Tee-Object -FilePath (Join-Path $Ev "golden.log") | Select-Object -Last 2

Write-Host "== 4/5 live stream (30 windows) ==" -ForegroundColor Cyan
python (Join-Path $Root "stm32_deployment\tools\live_stream.py") --port $Port --n 30 --flash 2>&1 |
    Tee-Object -FilePath (Join-Path $Ev "stream.log") | Select-Object -Last 3

Write-Host "== 5/5 report ==" -ForegroundColor Cyan
python (Join-Path $Root "stm32_deployment\tools\gen_report.py")
Start-Process (Join-Path $Root "docs\REPORT.html")
