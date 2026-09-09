param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-WarnLine {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Find-Python {
    # Prefer Python 3.11 because MemeMorph CI uses it.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.11 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{
                    Command = "py"
                    Args = @("-3.11")
                }
            }
        }
        catch {
            # Fall through to normal python.
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{
            Command = "python"
            Args = @()
        }
    }

    throw "Python was not found. Install Python 3.11+ and run this script again."
}

function Invoke-Python {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command $($Arguments -join ' ')"
    }
}

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Label
    )

    if (Test-Path $Destination) {
        $size = (Get-Item $Destination).Length

        if ($size -gt 1024) {
            Write-Success "$Label already exists."
            return
        }

        Write-WarnLine "$Label exists but looks incomplete. Re-downloading."
        Remove-Item $Destination -Force
    }

    Write-Host "Downloading $Label..."
    Invoke-WebRequest `
        -Uri $Url `
        -OutFile $Destination `
        -UseBasicParsing

    if (-not (Test-Path $Destination)) {
        throw "Download failed for $Label."
    }

    $downloadedSize = (Get-Item $Destination).Length

    if ($downloadedSize -le 1024) {
        throw "$Label downloaded, but the file is unexpectedly small."
    }

    Write-Success "$Label downloaded."
}


# ------------------------------------------------------------
# Resolve project paths
# ------------------------------------------------------------

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDirectory

Set-Location $ProjectRoot

Write-Host ""
Write-Host "MemeMorph Setup" -ForegroundColor Magenta
Write-Host "==============="
Write-Host "Project root: $ProjectRoot"


# ------------------------------------------------------------
# Check Python
# ------------------------------------------------------------

Write-Step "Checking Python"

$PythonLauncher = Find-Python

$pythonVersionArgs = @()
$pythonVersionArgs += $PythonLauncher.Args
$pythonVersionArgs += "--version"

& $PythonLauncher.Command @pythonVersionArgs

if ($LASTEXITCODE -ne 0) {
    throw "Unable to run Python."
}

Write-Success "Python is available."


# ------------------------------------------------------------
# Create virtual environment
# ------------------------------------------------------------

Write-Step "Creating virtual environment"

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    $venvArgs = @()
    $venvArgs += $PythonLauncher.Args
    $venvArgs += @("-m", "venv", ".venv")

    Invoke-Python `
        -Command $PythonLauncher.Command `
        -Arguments $venvArgs

    Write-Success "Created .venv."
}
else {
    Write-Success ".venv already exists."
}


# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

Write-Step "Installing Python dependencies"

Invoke-Python `
    -Command $VenvPython `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    )

Invoke-Python `
    -Command $VenvPython `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt"
    )

Write-Success "Python dependencies installed."


# ------------------------------------------------------------
# Create required directories
# ------------------------------------------------------------

Write-Step "Creating required directories"

$ModelsDirectory = Join-Path $ProjectRoot "models"
$ReactionDirectory = Join-Path $ProjectRoot "assets\reactions"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $ModelsDirectory `
    | Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path $ReactionDirectory `
    | Out-Null

Write-Success "Required directories are ready."


# ------------------------------------------------------------
# Download MediaPipe models
# ------------------------------------------------------------

Write-Step "Downloading MediaPipe models"

$Models = @(
    @{
        Label = "Face Landmarker"
        File = "face_landmarker.task"
        Url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    },
    @{
        Label = "Hand Landmarker"
        File = "hand_landmarker.task"
        Url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    },
    @{
        Label = "Pose Landmarker Lite"
        File = "pose_landmarker_lite.task"
        Url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    }
)

foreach ($Model in $Models) {
    $Destination = Join-Path $ModelsDirectory $Model.File

    Download-IfMissing `
        -Url $Model.Url `
        -Destination $Destination `
        -Label $Model.Label
}


# ------------------------------------------------------------
# Verify reaction media directory
# ------------------------------------------------------------

Write-Step "Checking reaction media"

$ExpectedReactionFiles = @(
    "speed_reverse_smile.gif",
    "rock_eyebrow.gif",
    "surprised_pikachu.gif",
    "sad_cry.gif",
    "side_eye.gif",
    "mocking_tom_jerry.gif",
    "laughing.gif",
    "hands_on_head.gif",
    "thumbs_up.gif",
    "absolute_cinema.gif"
)

$MissingReactionFiles = @()

foreach ($File in $ExpectedReactionFiles) {
    $Path = Join-Path $ReactionDirectory $File

    if (-not (Test-Path $Path)) {
        $MissingReactionFiles += $File
    }
}

if ($MissingReactionFiles.Count -eq 0) {
    Write-Success "All reaction media files are present."
}
else {
    Write-WarnLine "Some reaction media files are missing."
    Write-Host ""
    Write-Host "MemeMorph does not download third-party meme GIFs automatically."
    Write-Host "Add your own media to:"
    Write-Host "  assets\reactions\"
    Write-Host ""
    Write-Host "Expected filenames:"

    foreach ($File in $MissingReactionFiles) {
        Write-Host "  - $File"
    }
}


# ------------------------------------------------------------
# Check Azure configuration
# ------------------------------------------------------------

Write-Step "Checking configuration mode"

if ($Env:AZURE_APPCONFIG_ENDPOINT) {
    Write-Success "AZURE_APPCONFIG_ENDPOINT is set."
    Write-Host "MemeMorph will attempt Azure App Configuration at startup."
}
else {
    Write-Success "Azure endpoint not set. Local configuration fallback will be used."
}


# ------------------------------------------------------------
# Run tests
# ------------------------------------------------------------

if (-not $SkipTests) {
    Write-Step "Running automated tests"

    Invoke-Python `
        -Command $VenvPython `
        -Arguments @(
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v"
        )

    Write-Success "Automated tests passed."
}
else {
    Write-WarnLine "Tests skipped because -SkipTests was supplied."
}


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "MemeMorph setup completed successfully." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Run MemeMorph with:"
Write-Host ""
Write-Host "  .\.venv\Scripts\python.exe src\main.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Optional: activate the environment first:"
Write-Host ""
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python src\main.py"
Write-Host ""
