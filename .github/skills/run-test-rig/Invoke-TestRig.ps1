[CmdletBinding()]
param (
    [Parameter(Mandatory,
               HelpMessage="Duration in seconds for test on the test rig.")]
    [int]
    $DurationSeconds
)

$klipper_url = "klippy-test"

function Invoke-SshCommand {
    Write-Host -ForegroundColor Cyan "ssh $klipper_url $args"
    ssh $klipper_url $args
}

function Invoke-ScpCommand {
    Write-Host -ForegroundColor Cyan "scp $args"
    scp $args
}

$GCode = Get-Content "$PSScriptRoot\startup.gcode" -Encoding utf8

$found = $false
Invoke-SshCommand "cat printer_data/config/printer.cfg" |
    ForEach-Object {
        if ($found -and $_.StartsWith("#*#")) {
            $found = $false
            $_
        } elseif ($_.StartsWith("[delayed_gcode startup]")) {
            $found = $true

            $_
            "initial_duration: 1"
            "gcode:"
            "  SET_KINEMATIC_POSITION X=150 Y=150 Z=25 SET_HOMED=XYZ"
            "  MMU_TEST_CONFIG LOG_LEVEL=4"
            "  MMU_SELECT GATE=0"
            $GCode |
                ForEach-Object { $_.Split("`n") } |
                ForEach-Object { "  $_" }
            ""
        } elseif (!$found) {
            $_
        }
    } |
    Set-Content "$env:temp\printer.cfg"

Invoke-ScpCommand "$env:temp\printer.cfg" "${klipper_url}:printer_data/config/printer.cfg"

Invoke-SshCommand 'pgrep -f klippy-env/bin/python && kill `pgrep -f klippy-env/bin/python`'
Invoke-SshCommand 'rm /home/user/printer_data/logs/klippy.log'

$p = Start-Process `
    -NoNewWindow `
    -PassThru `
    -FilePath "ssh" `
    -ArgumentList @(
        $klipper_url
        "/home/user/klippy-env/bin/python /home/user/klipper/klippy/klippy.py /home/user/printer_data/config/printer.cfg -I /home/user/printer_data/comms/klippy.serial -l /home/user/printer_data/logs/klippy.log -a /home/user/printer_data/comms/klippy.sock;"
    )`
    -RedirectStandardOutput "$env:temp\scp_output.txt" `
    -RedirectStandardError "$env:temp\scp_error.txt"
if (!$p.WaitForExit($DurationSeconds * 1000)) {
    $p.Kill()
    Invoke-SshCommand 'pgrep -f klippy-env/bin/python && kill `pgrep -f klippy-env/bin/python`'
}

Invoke-ScpCommand "${klipper_url}:/home/user/printer_data/logs/klippy.log" "C:\git\YAMMU\Firmware\Happy-Hare\.github\skills\run-test-rig\klippy.log"

Write-Host "Test rig process exited with code $($p.ExitCode). Output:"
Get-Content "$env:temp\scp_output.txt" | ForEach-Object { Write-Host $_ }
Get-Content "$env:temp\scp_error.txt" | ForEach-Object { Write-Host $_ }

