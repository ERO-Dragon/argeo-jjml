#requires -Version 5.1
<#
.SYNOPSIS
    Verifies that the JJML artifacts under LlamaDependencies can be loaded by a Java 17 runtime.

.DESCRIPTION
    A jar compiled with a JDK newer than the target runtime still looks fine to javac:
    --release only constrains the classes being produced in that invocation, never the
    binaries already on the classpath. Handing over a Java 21 build to a Java 17 process
    therefore fails at class-load time with UnsupportedClassVersionError, not at build time.

    This script checks the produced artifacts instead of trusting the build log:
      * every class file in every jar is at most major version 61 (Java 17)
      * the OSGi requirement in the manifest is JavaSE-17 or lower
      * the expected JNI libraries are present
      * SHA-256 of every artifact, so a hand-over can be pinned

.EXAMPLE
    pwsh -File sdk/verify-jdk17-artifacts.ps1
    pwsh -File sdk/verify-jdk17-artifacts.ps1 -ArtifactDir D:/Minecraft/Tianshu_libs/libs
#>
[CmdletBinding()]
param(
    # Directory containing jar/ and native/ subdirectories.
    [string]$ArtifactDir = (Join-Path $PSScriptRoot '..\LlamaDependencies'),
    # 61 = Java 17. 65 = Java 21.
    [int]$MaxClassVersion = 61,
    [int]$MaxJavaRelease = 17
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

if (-not (Test-Path -LiteralPath $ArtifactDir)) {
    throw "Artifact directory not found: $ArtifactDir"
}
$ArtifactDir = (Resolve-Path -LiteralPath $ArtifactDir).Path

$jarDir = Join-Path $ArtifactDir 'jar'
$nativeDir = Join-Path $ArtifactDir 'native'
$jars = @(Get-ChildItem -LiteralPath $jarDir -Filter *.jar -File -ErrorAction SilentlyContinue)
if ($jars.Count -eq 0) {
    throw "No jars found under $jarDir"
}

$failures = New-Object 'System.Collections.Generic.List[string]'
$buffer = New-Object byte[] 8

function Get-JarManifest([string]$Path) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $entry = $zip.GetEntry('META-INF/MANIFEST.MF')
        if ($null -eq $entry) { return @{} }
        $reader = New-Object System.IO.StreamReader($entry.Open())
        try { $lines = $reader.ReadToEnd() -split "`r?`n" } finally { $reader.Dispose() }
    } finally {
        $zip.Dispose()
    }

    $headers = @{}
    $lastKey = $null
    foreach ($line in $lines) {
        if ($line -eq '') { continue }
        if ($line.StartsWith(' ') -and $lastKey) {
            $headers[$lastKey] += $line.Substring(1)
            continue
        }
        $idx = $line.IndexOf(':')
        if ($idx -lt 1) { continue }
        $lastKey = $line.Substring(0, $idx).Trim()
        $headers[$lastKey] = $line.Substring($idx + 1).Trim()
    }
    return $headers
}

Write-Host "JJML artifact check -> $ArtifactDir" -ForegroundColor Cyan
Write-Host ("Allowed class file version: {0} (Java {1})" -f $MaxClassVersion, $MaxJavaRelease)
Write-Host ''

foreach ($jar in $jars) {
    Write-Host ("[{0}]" -f $jar.Name) -ForegroundColor Cyan

    $zip = [System.IO.Compression.ZipFile]::OpenRead($jar.FullName)
    try {
        $count = 0
        $maxMajor = 0
        $offenders = New-Object 'System.Collections.Generic.List[string]'

        foreach ($entry in $zip.Entries) {
            if (-not $entry.FullName.EndsWith('.class')) { continue }
            $stream = $entry.Open()
            try { $read = $stream.Read($buffer, 0, 8) } finally { $stream.Dispose() }
            if ($read -lt 8) { continue }

            $major = ([int]$buffer[6] -shl 8) -bor [int]$buffer[7]
            $count++
            if ($major -gt $maxMajor) { $maxMajor = $major }
            if ($major -gt $MaxClassVersion) { $offenders.Add("$($entry.FullName)=$major") }
        }
    } finally {
        $zip.Dispose()
    }

    $versionOk = ($count -gt 0 -and $offenders.Count -eq 0)
    if ($versionOk) {
        Write-Host ("  class version   : OK (max {0}, {1} classes)" -f $maxMajor, $count) -ForegroundColor Green
    } else {
        Write-Host ("  class version   : FAIL (max {0}); {1} class(es) exceed {2}" -f $maxMajor, $offenders.Count, $MaxClassVersion) -ForegroundColor Red
        foreach ($o in ($offenders | Select-Object -First 10)) { Write-Host "      $o" -ForegroundColor Red }
        $failures.Add("$($jar.Name): class file version $maxMajor > $MaxClassVersion")
    }

    $manifest = Get-JarManifest $jar.FullName
    $bree = $manifest['Bundle-RequiredExecutionEnvironment']
    if ($bree -match 'JavaSE-(\d+)') {
        $release = [int]$Matches[1]
        if ($release -le $MaxJavaRelease) {
            Write-Host ("  manifest BREE   : OK ({0})" -f $bree) -ForegroundColor Green
        } else {
            Write-Host ("  manifest BREE   : FAIL ({0})" -f $bree) -ForegroundColor Red
            $failures.Add("$($jar.Name): Bundle-RequiredExecutionEnvironment is $bree")
        }
    } else {
        Write-Host '  manifest BREE   : absent'
    }

    if ($manifest['Created-By']) {
        Write-Host ("  created by      : {0}" -f $manifest['Created-By'])
    }
    Write-Host ''
}

# JNI libraries: the Java side of JJML binds to these, so a half-copied set fails at runtime.
$expectedNative = @(
    'Java_org_argeo_jjml_ggml.dll',
    'Java_org_argeo_jjml_llm.dll',
    'Java_org_argeo_jjml_mtmd.dll',
    'Java_org_argeo_jjml_whisper.dll',
    'ggml-base.dll',
    'ggml.dll',
    'llama.dll'
)
if (Test-Path -LiteralPath $nativeDir) {
    Write-Host ("[native]") -ForegroundColor Cyan
    $present = @(Get-ChildItem -LiteralPath $nativeDir -Filter *.dll -File | Select-Object -ExpandProperty Name)
    Write-Host ("  {0} DLL(s) present" -f $present.Count)
    foreach ($name in $expectedNative) {
        if ($present -contains $name) {
            Write-Host ("  OK      {0}" -f $name) -ForegroundColor Green
        } else {
            Write-Host ("  MISSING {0}" -f $name) -ForegroundColor Red
            $failures.Add("native: missing $name")
        }
    }
    Write-Host ''
} else {
    Write-Host "No native/ directory under $ArtifactDir"
    Write-Host ''
}

Write-Host '[sha256]' -ForegroundColor Cyan
Get-ChildItem -LiteralPath $ArtifactDir -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host ("  {0}  {1}" -f $hash, $_.FullName.Substring($ArtifactDir.Length + 1))
    }
Write-Host ''

if ($failures.Count -gt 0) {
    Write-Host ("RESULT: FAIL ({0} problem(s))" -f $failures.Count) -ForegroundColor Red
    foreach ($f in $failures) { Write-Host "  - $f" -ForegroundColor Red }
    exit 1
}

Write-Host 'RESULT: OK - artifacts are loadable on a Java 17 runtime' -ForegroundColor Green
exit 0