# One-click verification: host parity -> build -> flash+boot -> golden -> replay -> stream -> latency -> report.

#

# Usage:

#   powershell -ExecutionPolicy Bypass -File stm32_deployment\tools\verify_all.ps1

#   powershell ... verify_all.ps1 -Port COM5 -LimitHost 4000 -SkipReport

#

# Each stage writes its log under stm32_deployment\results\evidence\.

# A machine-readable summary lands in stm32_deployment\results\verify_last.json

# (schema dsc-amcnet-verify-v1); exit code 0 only when every stage passed.

param(

    [string]$Port = "COM3",

    [int]$LimitHost = 0,        # 0 = all 44,000 locked-test windows

    [switch]$SkipHost,          # skip the host-side bit-exact parity stage

    [switch]$SkipReport,        # skip REPORT.html + figs regeneration

    [string]$OutDir = ""

)



$ErrorActionPreference = "Continue"

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # ...\dsc-amcnet-cortex-m7-amc

$Proj   = "D:\DSC_AMCNet_Deployment\MDK-ARM\DSC_AMCNet_Deployment.uvprojx"

$UV4    = "D:\APP\Keil5\UV4\UV4.exe"

$VenvPy = "D:\APP\Pytorch\Scripts\python.exe"   # zig + onnx + onnxruntime + pyserial + matplotlib

$SysPy  = $VenvPy                               # 单一解释器，避免 PATH 歧义

$Tools  = Join-Path $Root "stm32_deployment\tools"

$Ev     = if ($OutDir) { $OutDir } else { Join-Path $Root "stm32_deployment\results\evidence" }

New-Item -ItemType Directory -Force $Ev | Out-Null



$stages = [ordered]@{}

function Stage([string]$Name, [bool]$Ok, [string]$Detail) {

    $script:stages[$Name] = [pscustomobject]@{ ok = $Ok; detail = $Detail }

    Write-Host ("  [{0}] {1} - {2}" -f $(if ($Ok) { " OK " } else { "FAIL" }), $Name, $Detail) `
        -ForegroundColor $(if ($Ok) { "Green" } else { "Red" })

}



function Invoke-Logged([string]$Exe, [string[]]$ToolArgs, [string]$LogPath) {

    $out = & $Exe @ToolArgs 2>&1

    $code = $LASTEXITCODE

    $out | Set-Content -Path $LogPath -Encoding UTF8

    return @{ code = $code; text = ($out -join "`n") }

}



function Get-First([string]$Text, [string]$Pattern, [int]$Group = 1) {

    $m = [regex]::Match($Text, $Pattern)

    if ($m.Success) { return $m.Groups[$Group].Value }

    return ""

}



Write-Host "== 0/8 preflight ==" -ForegroundColor Cyan

$commit = (& git -C $Root rev-parse --short HEAD 2>$null) -join ""

$missing = @()

foreach ($p in @($UV4, $Proj, $VenvPy)) { if (-not (Test-Path $p)) { $missing += $p } }

Stage "preflight" ($missing.Count -eq 0) ("HEAD=" + $commit + " port=" + $Port + $(if ($missing) { " MISSING: " + ($missing -join ",") } else { "" }))



# --------------------------------------------------------------- host parity

$hostBit = ""; $hostAcc = ""

if (-not $SkipHost) {

    Write-Host "== 1/8 host-side bit-exact parity ==" -ForegroundColor Cyan

    $toolArgs = @((Join-Path $Tools "host_sim\compare_engine.py"))

    if ($LimitHost -gt 0) { $toolArgs += @("--limit", "$LimitHost") }

    $toolArgs += @("--json", (Join-Path $Root "stm32_deployment\results\host_parity_last.json"))

    $r = Invoke-Logged $VenvPy $toolArgs (Join-Path $Ev "host_parity.log")

    $hpJson = Join-Path $Root "stm32_deployment/results/host_parity_last.json"
    $hp = if (Test-Path $hpJson) { Get-Content $hpJson -Raw | ConvertFrom-Json } else { $null }

    if ($hp) {

        $hostBit = "{0:N3}" -f ($hp.bit_exact_fraction * 100)

        $hostAcc = "{0:N3}" -f ($hp.argmax_agree * 100)

    }

    Stage "host parity" (($r.code -eq 0) -and $hp -and ($hp.bit_exact_fraction -gt 0.95) -and ($hp.argmax_agree -gt 0.99)) `
        "n=$($hp.n_windows) bit-exact=$hostBit% argmax=$hostAcc% (exit=$($r.code))"

} else {

    Stage "host parity" $true "skipped"

}



# -------------------------------------------------------------------- build

Write-Host "== 2/8 build (incremental) ==" -ForegroundColor Cyan

$buildLog = Join-Path $Ev "build.log"

$r = Invoke-Logged $UV4 @("-b", $Proj, "-j0", "-o", $buildLog) $buildLog

$buildTxt = Get-Content $buildLog -Raw -ErrorAction SilentlyContinue

$errors = Get-First $buildTxt "(\d+)\s+Error"

$warns  = Get-First $buildTxt "(\d+)\s+Warning"

$codeSz = Get-First $buildTxt "Code=(\d+)"

$buildOk = $buildTxt -match "Build Time Elapsed"

Stage "build" ($buildOk -and ($errors -eq "0")) "Code=${codeSz}B errors=$errors warnings=$warns"



# ------------------------------------------------------ flash + boot capture

Write-Host "== 3/8 flash + boot capture (550 MHz bench) ==" -ForegroundColor Cyan

Get-Process UV4 -ErrorAction SilentlyContinue | Stop-Process -Force

$lat0 = ""; $latReal = ""; $bootOk = $false

$serial = New-Object System.IO.Ports.SerialPort $Port,115200,None,8,One

try {

    $serial.Open()

    $sb = New-Object System.Text.StringBuilder

    $flashLog = Join-Path $Ev "flash.log"

    $flashProc = Start-Process $UV4 -ArgumentList @("-f", $Proj, "-o", $flashLog) -PassThru

    $end = (Get-Date).AddSeconds(22)

    while ((Get-Date) -lt $end) {

        try { $s = $serial.ReadExisting(); if ($s) { [void]$sb.Append($s) } } catch {}

        Start-Sleep -Milliseconds 100

    }

    $null = $flashProc.WaitForExit(10000)

    $boot = $sb.ToString()

    $bootLines = ($boot -split "`r?`n" | Where-Object { $_ -match "\[DWT\]|\[AI\]|Cortex-M7 @|RESULT|BENCH" })

    $bootLines | Set-Content -Path (Join-Path $Ev "serial_boot.log") -Encoding UTF8

    $boot | Set-Content -Path (Join-Path $Ev "serial_raw.log") -Encoding UTF8 -ErrorAction SilentlyContinue

    $verifyOk = (Test-Path $flashLog) -and ((Get-Content $flashLog -Raw) -match "Verify OK")

    $lat0    = Get-First $boot "\[DWT\] N=1000\s+mean=([0-9.]+)"

    $latReal = Get-First $boot "(?s)RESULT-SAMPLE.{0,400}?Mean\s*:\s*([0-9.]+)\s*ms"

    $bootOk  = ($boot -match "\[DWT\] N=1000") -and ($boot -match "\[AI\] weights=") -and $verifyOk

    Stage "flash+boot" $bootOk "zero=$lat0 ms real=$latReal ms verify_ok=$verifyOk"

    $serial.Close()

} catch {

    Stage "flash+boot" $false "serial: $($_.Exception.Message)"

} finally {

    if ($serial.IsOpen) { $serial.Close() }

}



# ------------------------------------------------------------------- golden

Write-Host "== 4/8 golden vectors (22) ==" -ForegroundColor Cyan

$r = Invoke-Logged $SysPy @((Join-Path $Tools "golden_test.py"), "--port", $Port) (Join-Path $Ev "golden.log")

$goldPass = Get-First $r.text "(\d+)/22"

$goldOk = ($r.code -eq 0) -and ($goldPass -eq "22")

Stage "golden" $goldOk "passed=$goldPass/22 (exit=$($r.code))"



# ------------------------------------------------------------------- replay

Write-Host "== 5/8 replay 300 windows ==" -ForegroundColor Cyan

$r = Invoke-Logged $SysPy @((Join-Path $Tools "p0_replay.py"), "--n", "300", "--port", $Port) (Join-Path $Ev "replay.log")

$rpOk  = Get-First $r.text '"n_ok"\s*:\s*(\d+)'

$rpEx  = Get-First $r.text '"exact_11_logits"\s*:\s*(\d+)'

$rpArg = Get-First $r.text '"argmax_agree"\s*:\s*(\d+)'

Stage "replay" (($r.code -eq 0) -and ($rpOk -eq "300") -and ($rpArg -eq "300")) "n_ok=$rpOk exact=$rpEx argmax=$rpArg"



# ------------------------------------------------------------------- stream

Write-Host "== 6/8 live stream (20 windows) ==" -ForegroundColor Cyan

$r = Invoke-Logged $SysPy @((Join-Path $Tools "live_stream.py"), "--n", "20", "--port", $Port, "--flash") (Join-Path $Ev "stream.log")

$stWin = Get-First $r.text "windows=(\d+)/20"

$stAcc = Get-First $r.text "board_acc=([0-9.]+)"

$stUs  = Get-First $r.text "compute\s+mean=([0-9.]+)us"

Stage "stream" (($r.code -eq 0) -and ($stWin -eq "20")) "windows=$stWin/20 acc=$stAcc compute=${stUs}us"



# ------------------------------------------------------------------ latency

Write-Host "== 7/8 latency CSV (reset -> boot DWT -> docs/evidence) ==" -ForegroundColor Cyan

$csv = Join-Path $Root "docs\evidence\latency_dscamcnet_int8.csv"

$latProc = Start-Process $SysPy -ArgumentList @("-u", (Join-Path $Root "hardware_characterization\measure_latency.py"),

                   "--port", $Port, "--output", $csv, "--timeout", "20") `
                   -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $Ev "latency_stage.log") `
                   -RedirectStandardError (Join-Path $Ev "latency_stage.err")

Start-Sleep -Milliseconds 1200

$null = Start-Process $UV4 -ArgumentList @("-f", $Proj, "-o", (Join-Path $Ev "flash_latency.log")) -Wait -PassThru

$latProc | Wait-Process -Timeout 40 -ErrorAction SilentlyContinue

$rows = if (Test-Path $csv) { @(Get-Content $csv | Select-Object -Skip 1) } else { @() }

$csvOk = ($rows.Count -ge 2)

Stage "latency CSV" $csvOk "rows=$($rows.Count) first=$($rows[0])"



# ------------------------------------------------------------------- report

if (-not $SkipReport) {

    Write-Host "== 8/8 regenerate REPORT.html + figs ==" -ForegroundColor Cyan

    $r = Invoke-Logged $SysPy @((Join-Path $Root "stm32_deployment\tools\gen_report.py")) (Join-Path $Ev "gen_report.log")

    $repOk = Test-Path (Join-Path $Root "docs\REPORT.html")

    Stage "report" ($r.code -eq 0 -and $repOk) (Get-First $r.text "written:\s*(.+)")

} else {

    Stage "report" $true "skipped"

}



# ------------------------------------------------------------------ summary

$allOk = -not ($stages.Values | Where-Object { -not $_.ok })

$summary = [pscustomobject]@{

    schema     = "dsc-amcnet-verify-v1"

    generated  = (Get-Date).ToString("s")

    commit     = $commit

    port       = $Port

    all_passed = $allOk

    latency    = [pscustomobject]@{ zero_ms = $lat0; real_ms = $latReal }

    stages     = $stages

}

$jsonPath = Join-Path $Root "stm32_deployment\results\verify_last.json"

$summary | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8

Write-Host ""

Write-Host ("== RESULT: {0} ==" -f $(if ($allOk) { "ALL PASS" } else { "FAILED" })) -ForegroundColor $(if ($allOk) { "Green" } else { "Red" })

Write-Host "summary: $jsonPath"

exit $(if ($allOk) { 0 } else { 1 })
