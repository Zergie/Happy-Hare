[CmdletBinding()]
param (
    [Parameter(Mandatory,
               HelpMessage="Duration in seconds for test on the test rig.")]
    [int]
    $DurationSeconds
)

$pythonScript = Join-Path $PSScriptRoot "invoke_test_rig.py"

Write-Host -ForegroundColor Cyan "python $pythonScript -DurationSeconds $DurationSeconds"
python $pythonScript -DurationSeconds $DurationSeconds
exit $LASTEXITCODE

