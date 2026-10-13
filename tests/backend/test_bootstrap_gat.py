import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from app.bootstrap_gat import restore


class RestoreGATTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.target = self.root / 'restored'
        self.files = {
            'active.json': b'{"model_id":"saved-model"}',
            'datasets/dataset.json': b'{"sequence_dir":"/unavailable/old/sequences"}',
            'sequences/dataset/example.bin': b'preserved dataset bytes',
            'weights/model.pt': b'preserved model bytes',
        }
        archive = self.bundle / 'snapshot.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, value in self.files.items():
                member = tarfile.TarInfo(name)
                member.size = len(value)
                tar.addfile(member, io.BytesIO(value))
        self.manifest = {'dataset_id': 'dataset', 'archive': archive.name,
                         'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                         'files': [{'path': name, 'size': len(value),
                                    'sha256': hashlib.sha256(value).hexdigest()}
                                   for name, value in self.files.items()]}
        self.save_manifest()

    def save_manifest(self):
        (self.bundle / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_restore_relocate_and_preserve_local_activation(self):
        restore(self.bundle, self.target)
        self.assertEqual((self.target / 'weights/model.pt').read_bytes(), self.files['weights/model.pt'])
        data = json.loads((self.target / 'datasets/dataset.json').read_text())
        self.assertEqual(Path(data['sequence_dir']), (self.target / 'sequences/dataset').resolve())
        (self.target / 'active.json').write_text('{"model_id":null}')
        (self.target / 'weights/model.pt').unlink()
        restore(self.bundle, self.target)
        self.assertEqual(json.loads((self.target / 'active.json').read_text()), {'model_id': None})
        restore(self.bundle, self.target)

    def test_lfs_pointer_rejected(self):
        (self.bundle / 'snapshot.tar.gz').write_text('version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            restore(self.bundle, self.target)
        self.assertFalse(self.target.exists())

    def test_unsafe_path_rejected(self):
        self.manifest['files'][0]['path'] = '../outside'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Ruta invalida'):
            restore(self.bundle, self.target)

    def test_invalid_file_checksum_rejected(self):
        self.manifest['files'][0]['sha256'] = '0' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Archivo corrupto'):
            restore(self.bundle, self.target)
        self.assertFalse((self.target / 'active.json').exists())


if __name__ == '__main__':
    unittest.main()
