"""Empaqueta el dataset y el modelo GAT-LSTM activo para Git LFS."""
import hashlib
import json
from pathlib import Path
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import MODELS_DIR, artifact_directory, artifact_file


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = MODELS_DIR
    active = json.loads((source / 'active.json').read_text())
    model_id = active['model_id']
    model = json.loads((source / 'models' / f'{model_id}.json').read_text())
    if model.get('method') != 'gat_lstm' or model.get('config', {}).get('architecture_version') != 'gat_lstm_v3_segments_multiscale':
        raise ValueError('El modelo activo no es SegmentGATLSTM')
    dataset_id = model['dataset_id']
    files = ['active.json', f'datasets/{dataset_id}.json']
    files += [f'{folder}/{model_id}{suffix}' for folder, suffix in
              [('models', '.json'), ('weights', '.pt'), ('graphs', '.json'), ('preprocessors', '.json')]]
    sequence_root = artifact_directory(source, 'sequences')
    files += [f'sequences/{path.relative_to(sequence_root).as_posix()}'
              for path in sorted((sequence_root / dataset_id).rglob('*')) if path.is_file()]
    weights = source / 'weights' / f'{model_id}.pt'
    if digest(weights) != model['weights_sha256']:
        raise ValueError('Los pesos no coinciden con el modelo')
    output = root / 'artifacts/gat_lstm'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'snapshot.tar.gz'
    temporary = archive.with_suffix('.gz.tmp')
    inventory = []
    with tarfile.open(temporary, 'w:gz', compresslevel=6) as tar:
        for relative in files:
            path = artifact_file(source, relative)
            checksum = digest(path)
            print(f'Empaquetando {relative} ({path.stat().st_size} bytes)', flush=True)
            tar.add(path, arcname=relative, recursive=False)
            if digest(path) != checksum:
                raise RuntimeError(f'El archivo cambio durante el empaquetado: {path}')
            inventory.append({'path': relative, 'size': path.stat().st_size, 'sha256': checksum})
    temporary.replace(archive)
    manifest = {'version': 1, 'model_id': model_id, 'dataset_id': dataset_id,
                'archive': archive.name, 'sha256': digest(archive),
                'compressed_bytes': archive.stat().st_size, 'files': inventory}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Paquete listo: {archive.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    main()
