param(
    [string]$InputDir = 'storage/gat-mixto/inputs',
    [string]$RunDir = 'storage/checkpoints/mixto-01',
    [string]$PythonPath = '.venv-cuda/Scripts/python.exe',
    [string]$Device = 'cuda',
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 30,
    [int]$Patience = 5,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "No existe el Python del entorno: $PythonPath"
    }
    $sourcePaths = @((Join-Path $InputDir 'real.csv'), (Join-Path $InputDir 'sintetico.csv'))
    foreach ($sourcePath in $sourcePaths) {
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "Falta el CSV de entrada: $sourcePath. Extrae entradas-mixto.zip en la raiz de Artifact."
        }
    }
    & $PythonPath scripts/gat_lstm_check.py --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'El dispositivo no paso la comprobacion GAT-LSTM.' }
    $datasetResult = Join-Path $RunDir 'dataset.json'
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $datasetResult -PathType Leaf)) {
            throw 'No hay dataset preparado para reanudar; ejecuta primero sin -Resume.'
        }
    } else {
        if (Test-Path -LiteralPath $datasetResult) {
            throw 'La ejecucion ya tiene un dataset; usa -Resume o cambia -RunDir.'
        }
        & $PythonPath scripts/gat_lstm_prepare.py --csv $sourcePaths[0] --csv $sourcePaths[1] `
            --config training/config-mixto.json --development-start '2026-02-01T00:00:00Z' `
            --development-end '2026-05-21T00:00:00Z' --result $datasetResult
        if ($LASTEXITCODE -ne 0) { throw 'No se pudo preparar un dataset entrenable con ambos CSV.' }
    }
    $prepared = Get-Content -Raw -LiteralPath $datasetResult | ConvertFrom-Json
    if (-not $prepared.ready_for_training) { throw 'El dataset no tiene soporte supervisado suficiente.' }
    foreach ($sourcePath in $sourcePaths) {
        $fileName = Split-Path -Leaf $sourcePath
        $expected = @($prepared.source_inputs | Where-Object { $_.filename -eq $fileName })
        $actual = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected.Count -ne 1 -or $expected[0].sha256 -ne $actual) {
            throw "El CSV $fileName cambio desde la preparacion; inicia otra ejecucion."
        }
    }
    $trainingArguments = @('scripts/gat_lstm_train.py', $prepared.dataset_id, '--device', $Device,
        '--batch-size', $BatchSize, '--max-epochs', $MaxEpochs, '--patience', $Patience,
        '--checkpoint-dir', $RunDir, '--result', (Join-Path $RunDir 'result.json'))
    if ($Resume -and (Test-Path -LiteralPath (Join-Path $RunDir 'run.json'))) {
        $trainingArguments += '--resume'
    }
    & $PythonPath @trainingArguments
    if ($LASTEXITCODE -ne 0) { throw 'Entrenamiento incompleto; revisa el error y usa -Resume para continuar.' }
    Write-Host "Candidato listo. Resultado: $(Join-Path $RunDir 'result.json')"
} finally {
    Pop-Location
}
