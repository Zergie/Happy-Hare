$pythonScript = Join-Path $PSScriptRoot "export_to_test_rig.py"

Write-Host -ForegroundColor Cyan "python $pythonScript"
python $pythonScript
exit $LASTEXITCODE