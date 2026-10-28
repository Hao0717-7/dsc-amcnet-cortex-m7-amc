<#
    build_flash.ps1 — build the Keil project, flash it over ST-Link and capture
    the boot banner / benchmark lines from the board's VCP.

    Usage:
        .\build_flash.ps1                 # build + flash + 25 s capture
        .\build_flash.ps1 -SkipBuild      # flash only
        .\build_flash.ps1 -Port COM5
#>
param(
    [string]$Project = "D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx",
    [string]$Port = "COM3",
    [int]$CaptureSeconds = 25,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$uv = "D:\APP\Keil5\UV4\UV4.exe"
$logDir = Join-Path $env:TEMP "dsc_build_flash"
New-Item -ItemType Directory -Force $logDir | Out-Null

if (-not $SkipBuild) {
    $buildLog = Join-Path $logDir "build.log"
    Remove-Item $buildLog -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $uv -ArgumentList @("-b", $Project, "-j0", "-o", $buildLog) -Wait -PassThru
    $tail = Get-Content $buildLog | Select-String -Pattern "error:|Program Size" | Select-Object -Last 3
    $tail | ForEach-Object { $_.Line }
    if ($tail -match "error:") { throw "build failed" }
}

$port = New-Object System.IO.Ports.SerialPort $Port, 115200, None, 8, One
$port.ReadTimeout = 500
$port.Open()
$sb = New-Object System.Text.StringBuilder
$flashLog = Join-Path $logDir "flash.log"
Remove-Item $flashLog -ErrorAction SilentlyContinue
Start-Process -FilePath $uv -ArgumentList @("-f", $Project, "-o", $flashLog) | Out-Null
$end = (Get-Date).AddSeconds($CaptureSeconds)
while ((Get-Date) -lt $end) {
    try { $s = $port.ReadExisting(); if ($s) { [void]$sb.Append($s) } } catch {}
    Start-Sleep -Milliseconds 150
}
$port.Close()

$sb.ToString() -split "`r?`n" | Where-Object {
    $_ -match '\[PROF|\[DWT|\[RESULT|Mean|\[AI\]|\[REPLAY|\[UART\]'
} | Select-Object -First 20
