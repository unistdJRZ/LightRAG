# codex_proxy.ps1
chcp 65001 | Out-Null

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$Utf8Bom   = New-Object System.Text.UTF8Encoding $true

[Console]::InputEncoding  = $Utf8NoBom
[Console]::OutputEncoding = $Utf8Bom
$OutputEncoding = $Utf8NoBom

$env:HTTP_PROXY  = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = $env:HTTP_PROXY
$env:NO_PROXY    = "localhost,127.0.0.1,::1"

& "$env:AppData\npm\codex.cmd" @args

