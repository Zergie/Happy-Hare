Set-StrictMode -Version Latest

$script:RunTestRigRoot = Split-Path -Parent $PSScriptRoot
$script:StartupGCodePath = Join-Path $script:RunTestRigRoot 'startup.gcode'
$script:InvokeScriptPath = Join-Path $script:RunTestRigRoot 'Invoke-TestRig.ps1'
$script:ExportScriptPath = Join-Path $script:RunTestRigRoot 'Export-ToTestRig.ps1'
$script:KlippyLogPath = Join-Path $script:RunTestRigRoot 'klippy.log'
$script:TargetUnit = 'mmu_gear_bldc unit0'

function Backup-RunTestRigStartupGCode {
    if (Test-Path $script:StartupGCodePath) {
        return Get-Content -Path $script:StartupGCodePath -Raw -Encoding utf8
    }
    return ''
}

function Restore-RunTestRigStartupGCode {
    param(
        [AllowEmptyString()]
        [string]$Content
    )

    if ([string]::IsNullOrEmpty($Content)) {
        Set-Content -Path $script:StartupGCodePath -Value @() -Encoding utf8
        return
    }
    Set-Content -Path $script:StartupGCodePath -Value $Content -Encoding utf8
}

function Set-RunTestRigStartupGCode {
    param(
        [Parameter(Mandatory)]
        [string[]]$Lines
    )

    Set-Content -Path $script:StartupGCodePath -Value $Lines -Encoding utf8
}

function Get-RunTestRigDurationSeconds {
    param(
        [Parameter(Mandatory)]
        [double]$ExpectedRuntimeSeconds,
        [double]$StartupSeconds = 1.0,
        [double]$ExtraSeconds = 0.0,
        [int]$MinimumDurationSeconds = 5
    )

    $durationSeconds = [int][Math]::Ceiling(($ExpectedRuntimeSeconds + $StartupSeconds + $ExtraSeconds) * 1.5)
    return [Math]::Max($MinimumDurationSeconds, $durationSeconds)
}

function Invoke-RunTestRigExport {
    & $script:ExportScriptPath
}

function Invoke-RunTestRigScenario {
    param(
        [Parameter(Mandatory)]
        [string[]]$GCodeLines,
        [Parameter(Mandatory)]
        [double]$ExpectedRuntimeSeconds,
        [double]$ExtraSeconds = 0.0
    )

    Set-RunTestRigStartupGCode -Lines $GCodeLines
    $durationSeconds = Get-RunTestRigDurationSeconds -ExpectedRuntimeSeconds $ExpectedRuntimeSeconds -ExtraSeconds $ExtraSeconds
    $output = & $script:InvokeScriptPath -DurationSeconds $durationSeconds 2>&1 | Out-String
    if (-not (Test-Path $script:KlippyLogPath)) {
        throw "Missing klippy.log after Invoke-TestRig. Script output:`n$output"
    }

    return [pscustomobject]@{
        DurationSeconds = $durationSeconds
        Output = $output
        LogText = Get-Content -Path $script:KlippyLogPath -Raw -Encoding utf8
        LogPath = $script:KlippyLogPath
    }
}

function Get-RunTestRigFatalLines {
    param(
        [Parameter(Mandatory)]
        [string]$LogText
    )

    return [regex]::Matches($LogText, '(?m)^.*(?:Unknown command:|Traceback \(most recent call last\)).*$') |
        ForEach-Object { $_.Value.Trim() }
}

function Get-BldcTachEntries {
    param(
        [Parameter(Mandatory)]
        [string]$LogText,
        [string]$Unit = $script:TargetUnit
    )

    $matches = [regex]::Matches($LogText, '(?m)^.*BLDC_TACH: freq=(?<freq>[0-9.]+) rpm=(?<rpm>[0-9.]+) unit=(?<unit>.+)$')
    $entries = @()
    foreach ($match in $matches) {
        if ($match.Groups['unit'].Value.Trim() -ne $Unit) {
            continue
        }
        $entries += [pscustomobject]@{
            Line = $match.Value.Trim()
            Frequency = [double]$match.Groups['freq'].Value
            Rpm = [double]$match.Groups['rpm'].Value
            Unit = $match.Groups['unit'].Value.Trim()
        }
    }
    return $entries
}

function Get-BldcPinEvents {
    param(
        [Parameter(Mandatory)]
        [string]$LogText,
        [string]$Unit = $script:TargetUnit
    )

    $matches = [regex]::Matches($LogText, '(?m)^.*BLDC_SET_PIN: (?<message>.+?) applied=(?<applied>[0-9.]+)(?: time=(?<time>[0-9.]+))?.* unit=(?<unit>.+)$')
    $events = @()
    foreach ($match in $matches) {
        if ($match.Groups['unit'].Value.Trim() -ne $Unit) {
            continue
        }
        $timeValue = $null
        if ($match.Groups['time'].Success -and $match.Groups['time'].Value) {
            $timeValue = [double]$match.Groups['time'].Value
        }
        $events += [pscustomobject]@{
            Line = $match.Value.Trim()
            Message = $match.Groups['message'].Value.Trim()
            Applied = [double]$match.Groups['applied'].Value
            Time = $timeValue
            Unit = $match.Groups['unit'].Value.Trim()
        }
    }
    return $events
}

function Get-BldcRuntimeSeconds {
    param(
        [Parameter(Mandatory)]
        [string]$LogText,
        [string]$Unit = $script:TargetUnit
    )

    $events = @(Get-BldcPinEvents -LogText $LogText -Unit $Unit | Where-Object { $_.Time -ne $null })
    $startEvent = $events | Where-Object { $_.Applied -gt 0.0 } | Select-Object -First 1
    if ($null -eq $startEvent) {
        throw "No non-zero BLDC_SET_PIN event with time found for unit '$Unit'."
    }

    $stopEvent = $events |
        Where-Object { $_.Time -gt $startEvent.Time -and $_.Applied -le 0.0 } |
        Select-Object -First 1
    if ($null -eq $stopEvent) {
        throw "No zero BLDC_SET_PIN event after first non-zero event found for unit '$Unit'."
    }

    return [pscustomobject]@{
        RuntimeSeconds = [double]($stopEvent.Time - $startEvent.Time)
        StartEvent = $startEvent
        StopEvent = $stopEvent
    }
}

function Get-BldcObservedRpm {
    param(
        [Parameter(Mandatory)]
        [string]$LogText,
        [string]$Unit = $script:TargetUnit
    )

    $entries = @(Get-BldcTachEntries -LogText $LogText -Unit $Unit)
    if ($entries.Count -eq 0) {
        throw "No BLDC_TACH lines found for unit '$Unit'."
    }

    return ($entries | Measure-Object -Property Rpm -Maximum).Maximum
}

function Get-BldcEvidencePreview {
    param(
        [Parameter(Mandatory)]
        [string]$LogText,
        [string]$Unit = $script:TargetUnit,
        [int]$Count = 5
    )

    return [regex]::Matches($LogText, '(?m)^.*BLDC_(?:TACH|SET_PIN).*$') |
        ForEach-Object { $_.Value.Trim() } |
        Where-Object { $_ -match [regex]::Escape($Unit) } |
        Select-Object -First $Count
}

function Assert-BldcEvidencePresent {
    param(
        [Parameter(Mandatory)]
        [string]$LogText,
        [string]$Unit = $script:TargetUnit
    )

    $tachEntries = @(Get-BldcTachEntries -LogText $LogText -Unit $Unit)
    $pinEvents = @(Get-BldcPinEvents -LogText $LogText -Unit $Unit)
    if ($tachEntries.Count -eq 0 -or $pinEvents.Count -eq 0) {
        $preview = Get-BldcEvidencePreview -LogText $LogText -Unit $Unit
        throw "Missing BLDC evidence for unit '$Unit'. TACH=$($tachEntries.Count) SET_PIN=$($pinEvents.Count). Preview:`n$($preview -join "`n")"
    }
}

function Assert-ValueWithinTolerance {
    param(
        [Parameter(Mandatory)]
        [string]$Label,
        [Parameter(Mandatory)]
        [double]$Observed,
        [Parameter(Mandatory)]
        [double]$Expected,
        [Parameter(Mandatory)]
        [double]$Tolerance,
        [string[]]$EvidenceLines = @()
    )

    if ([Math]::Abs($Observed - $Expected) -le $Tolerance) {
        return
    }

    $evidenceBlock = ''
    if ($EvidenceLines.Count -gt 0) {
        $evidenceBlock = "`nEvidence:`n$($EvidenceLines -join "`n")"
    }
    throw "$Label mismatch. Expected $Expected +/- $Tolerance but observed $Observed.$evidenceBlock"
}

Export-ModuleMember -Function @(
    'Assert-BldcEvidencePresent',
    'Assert-RunTestRigHealthy',
    'Assert-ValueWithinTolerance',
    'Backup-RunTestRigStartupGCode',
    'Get-BldcEvidencePreview',
    'Get-BldcObservedRpm',
    'Get-BldcPinEvents',
    'Get-BldcRuntimeSeconds',
    'Get-BldcTachEntries',
    'Get-RunTestRigDurationSeconds',
    'Invoke-RunTestRigExport',
    'Invoke-RunTestRigScenario',
    'Restore-RunTestRigStartupGCode',
    'Set-RunTestRigStartupGCode'
)
