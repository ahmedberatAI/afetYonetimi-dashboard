param(
    [int]$ApiPort = 8787,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Frontend = Join-Path $PSScriptRoot "frontend"
$ApiUrl = "http://127.0.0.1:$ApiPort"
$WebUrl = "http://127.0.0.1:$WebPort"

Write-Host "Afet Yönetimi local web app başlatılıyor..."
Write-Host "API: $ApiUrl"
Write-Host "Web: $WebUrl"

$apiJob = Start-Job -Name "afet-api" -ScriptBlock {
    param($RootPath, $Port)
    Set-Location $RootPath
    python -m uvicorn webapp.backend.app.main:app --host 127.0.0.1 --port $Port --reload
} -ArgumentList $Root, $ApiPort

Start-Sleep -Seconds 2

$webJob = Start-Job -Name "afet-web" -ScriptBlock {
    param($FrontendPath, $ApiBase, $Port)
    Set-Location $FrontendPath
    $env:VITE_API_BASE = $ApiBase
    npm run dev -- --host 127.0.0.1 --port $Port
} -ArgumentList $Frontend, $ApiUrl, $WebPort

try {
    while ($apiJob.State -eq "Running" -and $webJob.State -eq "Running") {
        Receive-Job -Job $apiJob, $webJob
        Start-Sleep -Milliseconds 900
    }

    Receive-Job -Job $apiJob, $webJob
    throw "Local web app processlerinden biri durdu. API state=$($apiJob.State), Web state=$($webJob.State)"
}
finally {
    Stop-Job -Job $apiJob, $webJob -ErrorAction SilentlyContinue
    Remove-Job -Job $apiJob, $webJob -Force -ErrorAction SilentlyContinue
}
