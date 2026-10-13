# Portable entry points when GNU Make is not installed on Windows.
# Example: .\make.ps1 agentpoison --num-queries 2
param(
    [Parameter(Position=0)]
    [string]$Target = 'help',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$TaskArguments
)
$ErrorActionPreference = 'Stop'
$TaskPython = Join-Path $PSScriptRoot '.venv-adapt/Scripts/python.exe'
$TaskCommands = @{
    'aqua' = @('-m', 'src.defense.aqua')
    'aqua-smoke' = @('-m', 'src.defense.aqua', 'smoke')
    'aqua-prepare' = @('-m', 'src.defense.aqua', 'prepare')
    'aqua-collect' = @('-m', 'src.defense.aqua', 'collect')
    'aqua-train' = @('-m', 'src.defense.aqua', 'train')
    'aqua-evaluate' = @('-m', 'src.defense.aqua', 'evaluate')
    'trigger-hotflip-margin' = @('-m', 'src.attack.margin.hotflip_margin')
    'dilution' = @('-m', 'src.attack.dilution')
    'dilution-smoke' = @('-m', 'src.attack.dilution', 'smoke', '--fixture')
    'dilution-prepare' = @('-m', 'src.attack.dilution', 'prepare-episodes')
    'dilution-index' = @('-m', 'src.attack.dilution', 'index')
    'dilution-train' = @('-m', 'src.attack.dilution', 'train')
    'dilution-evaluate' = @('-m', 'src.attack.dilution', 'evaluate')
    'dilution-drift' = @('-m', 'src.attack.dilution', 'prepare-drift')
    'dilution-adapt' = @('-m', 'src.attack.dilution', 'adapt')
    'dilution-evaluate-drift' = @('-m', 'src.attack.dilution', 'evaluate-drift')
    'dilution-report' = @('-m', 'src.attack.dilution', 'report')
    'artemis-plan' = @('-m', 'src.artemis', '--plan')
    'artemis-fixture' = @('-m', 'src.artemis', '--backend', 'fixture')
    'artemis-live' = @('-m', 'src.artemis', '--backend', 'live')
    'gate-demo' = @('-m', 'src.artemis.run_gate_demo')
}
if ($Target -eq 'help') {
    Write-Output 'Usage: .\make.ps1 TARGET [Python CLI arguments]'
    Write-Output ($TaskCommands.Keys | Sort-Object)
    exit 0
}
if (-not $TaskCommands.ContainsKey($Target)) {
    throw "Unknown target: $Target. Run .\make.ps1 help. GNU Make supports additional upstream targets."
}
Push-Location $PSScriptRoot
try {
    $TaskCommand = $TaskCommands[$Target]
    & $TaskPython @TaskCommand @TaskArguments
    $TaskExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $TaskExit
