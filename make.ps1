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
    'trigger-suggest' = @('-m', 'src.triggers.specificity.suggest')
    'trigger-specificity' = @('-m', 'src.triggers.specificity')
    'trigger-language-audit' = @('-m', 'src.triggers.hierarchy.language_audit')
    'trigger-hierarchy' = @('-m', 'src.triggers.hierarchy', 'compare')
    'trigger-hierarchy-smoke' = @('-m', 'src.triggers.hierarchy', 'smoke')
    'aqua' = @('-m', 'src.aqua')
    'aqua-smoke' = @('-m', 'src.aqua', 'smoke')
    'aqua-prepare' = @('-m', 'src.aqua', 'prepare')
    'aqua-collect' = @('-m', 'src.aqua', 'collect')
    'aqua-train' = @('-m', 'src.aqua', 'train')
    'aqua-evaluate' = @('-m', 'src.aqua', 'evaluate')
    'agentpoison-check' = @('-m', 'src.agentpoison.strategyqa', 'check')
    'agentpoison-index' = @('-m', 'src.agentpoison.strategyqa', 'index')
    'agentpoison' = @('-m', 'src.agentpoison.strategyqa', 'run')
    'agentpoison-report' = @('-m', 'src.agentpoison.strategyqa', 'report')
    'agentpoison-demo' = @('-m', 'src.agentpoison.run_agentpoison_demo')
    'agentpoison-optimize' = @('-m', 'src.agentpoison.phases', 'optimize')
    'agentpoison-prepare' = @('-m', 'src.agentpoison.phases', 'prepare')
    'agentpoison-retrieve' = @('-m', 'src.agentpoison.phases', 'retrieve')
    'agentpoison-infer' = @('-m', 'src.agentpoison.phases', 'infer')
    'agentpoison-evaluate' = @('-m', 'src.agentpoison.phases', 'evaluate')
    'agentpoison-all' = @('-m', 'src.agentpoison.phases', 'all')
    'agentpoison-ablate' = @('-m', 'src.agentpoison.phases', 'ablate')
    'mcat' = @('-m', 'src.triggers.mcat')
    'mcat-smoke' = @('-m', 'src.triggers.mcat', 'smoke', '--fixture')
    'mcat-prepare' = @('-m', 'src.triggers.mcat', 'prepare-episodes')
    'mcat-index' = @('-m', 'src.triggers.mcat', 'index')
    'mcat-train' = @('-m', 'src.triggers.mcat', 'train')
    'mcat-evaluate' = @('-m', 'src.triggers.mcat', 'evaluate')
    'mcat-drift' = @('-m', 'src.triggers.mcat', 'prepare-drift')
    'mcat-adapt' = @('-m', 'src.triggers.mcat', 'adapt')
    'mcat-evaluate-drift' = @('-m', 'src.triggers.mcat', 'evaluate-drift')
    'mcat-report' = @('-m', 'src.triggers.mcat', 'report')
    'adapt-plan' = @('-m', 'src.adapt', '--plan')
    'adapt-fixture' = @('-m', 'src.adapt', '--backend', 'fixture')
    'adapt-live' = @('-m', 'src.adapt', '--backend', 'live')
    'gate-demo' = @('-m', 'src.adapt.run_gate_demo')
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
