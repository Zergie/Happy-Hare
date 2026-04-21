Set-StrictMode -Version Latest

Import-Module (Join-Path $PSScriptRoot 'Run-TestRig.Helpers.psm1') -Force

$runtimeToleranceSeconds = 0.11
$rpmToleranceRatio = 0.10
$targetUnit = 'mmu_gear_bldc unit0'

Describe 'Run-TestRig BLDC scenarios' {
    $originalStartupGCode = $null

    BeforeAll {
        $originalStartupGCode = Backup-RunTestRigStartupGCode
        # Invoke-RunTestRigExport
    }

    AfterAll {
        Restore-RunTestRigStartupGCode -Content $originalStartupGCode
    }

    # It 'Test 1: sync gear motor follows extruder move and runs exactly 2.0s' {
    #     $result = Invoke-RunTestRigScenario -GCodeLines @(
    #         'MMU_SYNC_GEAR_MOTOR SYNC=1',
    #         '_CLIENT_LINEAR_MOVE E=100 F=3000',
    #         'MMU_SYNC_GEAR_MOTOR SYNC=0'
    #     ) -ExpectedRuntimeSeconds 4.0

    #     Assert-BldcEvidencePresent -LogText $result.LogText -Unit $targetUnit

    #     $runtime = Get-BldcRuntimeSeconds -LogText $result.LogText -Unit $targetUnit
    #     Assert-ValueWithinTolerance `
    #         -Label 'Test 1 BLDC runtime' `
    #         -Observed $runtime.RuntimeSeconds `
    #         -Expected 2.0 `
    #         -Tolerance $runtimeToleranceSeconds `
    #         -EvidenceLines @($runtime.StartEvent.Line, $runtime.StopEvent.Line)
    # }

    It 'Test 2: MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200' {
        $result = Invoke-RunTestRigScenario -GCodeLines @(
            'MMU_TEST_MOVE MOTOR=gear MOVE=400 SPEED=200'
        ) -ExpectedRuntimeSeconds 3.0

        Assert-BldcEvidencePresent -LogText $result.LogText -Unit $targetUnit

        $runtime = Get-BldcRuntimeSeconds -LogText $result.LogText -Unit $targetUnit
        Assert-ValueWithinTolerance `
            -Label 'Test 2 BLDC runtime' `
            -Observed $runtime.RuntimeSeconds `
            -Expected 2.0 `
            -Tolerance $runtimeToleranceSeconds `
            -EvidenceLines @($runtime.StartEvent.Line, $runtime.StopEvent.Line)

        $observedRpm = Get-BldcObservedRpm -LogText $result.LogText -Unit $targetUnit
        $rpmTolerance = 3000 * $rpmToleranceRatio
        $tachPreview = @(Get-BldcTachEntries -LogText $result.LogText -Unit $targetUnit | Select-Object -First 5 | ForEach-Object { $_.Line })
        Assert-ValueWithinTolerance `
            -Label 'Test 2 BLDC rpm' `
            -Observed $observedRpm `
            -Expected 3000 `
            -Tolerance $rpmTolerance `
            -EvidenceLines $tachPreview
    }

    # It 'Test 3: synced move shows BLDC evidence, 0.25s runtime, and 6000 rpm' {
    #     $result = Invoke-RunTestRigScenario -GCodeLines @(
    #         'MMU_TEST_MOVE MOTOR=synced MOVE=400 SPEED=200'
    #     ) -ExpectedRuntimeSeconds 2.0

    #     Assert-BldcEvidencePresent -LogText $result.LogText -Unit $targetUnit

    #     $runtime = Get-BldcRuntimeSeconds -LogText $result.LogText -Unit $targetUnit
    #     Assert-ValueWithinTolerance `
    #         -Label 'Test 3 BLDC runtime' `
    #         -Observed $runtime.RuntimeSeconds `
    #         -Expected 1.0 `
    #         -Tolerance $runtimeToleranceSeconds `
    #         -EvidenceLines @($runtime.StartEvent.Line, $runtime.StopEvent.Line)

    #     $observedRpm = Get-BldcObservedRpm -LogText $result.LogText -Unit $targetUnit
    #     $rpmTolerance = $expectedMoveTestRpm * $rpmToleranceRatio
    #     $tachPreview = @(Get-BldcTachEntries -LogText $result.LogText -Unit $targetUnit | Select-Object -First 5 | ForEach-Object { $_.Line })
    #     Assert-ValueWithinTolerance `
    #         -Label 'Test 3 BLDC rpm' `
    #         -Observed $observedRpm `
    #         -Expected $expectedMoveTestRpm `
    #         -Tolerance $rpmTolerance `
    #         -EvidenceLines $tachPreview
    # }
}
