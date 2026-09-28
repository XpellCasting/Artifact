"""Restaura el paquete GAT-LSTM de Git LFS sin reconstruir ni entrenar."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relocate_dataset(destination, dataset_id):
    path = destination / 'datasets' / f'{dataset_id}.json'
    dataset = json.loads(path.read_text(encoding='utf-8'))
    sequences = (destination / 'sequences' / dataset_id).resolve()
    if not Path(dataset['sequence_dir']).is_dir() and sequences.is_dir():
        dataset['sequence_dir'] = str(sequences)
        temporary = path.with_suffix('.json.restore.tmp')
        temporary.write_text(json.dumps(dataset, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)


def restore(bundle: Path, destination: Path):
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    inventory = {}
    for entry in manifest['files']:
        name = entry['path']
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError(f'Ruta invalida en el paquete: {name}')
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f'Ruta fuera del destino: {name}')
        inventory[name] = entry
    # Existing installations retain their datasets, models and activation choice.
    missing = {name for name in inventory if not (destination / name).is_file()}
    if not missing:
        relocate_dataset(destination, manifest['dataset_id'])
        return
    archive = bundle / manifest['archive']
    if not archive.is_file() or digest(archive) != manifest['sha256']:
        raise ValueError('Paquete GAT-LSTM ausente o incompleto. Ejecuta git lfs pull.')
    print('Restaurando dataset y modelo GAT-LSTM guardados...', flush=True)
    restored = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name not in inventory or not member.isfile():
                raise ValueError(f'Entrada inesperada: {member.name}')
            if member.name not in missing:
                continue
            entry = inventory[member.name]
            if member.size != entry['size']:
                raise ValueError(f'Tamano incorrecto: {member.name}')
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restore.tmp')
            try:
                with tar.extractfile(member) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                if digest(temporary) != entry['sha256']:
                    raise ValueError(f'Archivo corrupto: {member.name}')
                temporary.replace(target)
                restored.add(member.name)
            finally:
                temporary.unlink(missing_ok=True)
    if restored != missing:
        raise ValueError('El paquete no contiene todos los archivos del manifiesto')
    relocate_dataset(destination, manifest['dataset_id'])
    print('Dataset y modelo disponibles; no se ejecuto entrenamiento.', flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=root / 'artifacts/gat_lstm')
    parser.add_argument('--destination', type=Path, default=Path(
        os.environ.get('REVIEW_DATA_DIR', str(root / 'data/review'))) / 'diagnostics')
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    restore(args.bundle, args.destination)
    if args.serve:
        os.execvp('uvicorn', ['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'])


if __name__ == '__main__':
    main()
