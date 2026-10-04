"""Append-only experiment records with content-addressed JSON artifacts."""
import json
import sqlite3
import os
import tempfile
from pathlib import Path
from .records import canonical, identity


class Store:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'objects').mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'experiments.sqlite3', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records '
                        '(id INTEGER PRIMARY KEY, kind TEXT, object_id TEXT, '
                        'created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
        self.db.execute('CREATE INDEX IF NOT EXISTS record_kind ON records(kind)')
        self.db.execute('CREATE TABLE IF NOT EXISTS experiment_index '
                        '(object_id TEXT PRIMARY KEY, model_id TEXT, workload_id TEXT, valid INTEGER)')

    def save(self, kind, record):
        key = identity(record)
        destination = self.root / 'objects' / (key + '.json')
        if not destination.exists():
            with tempfile.NamedTemporaryFile(mode='w',dir=destination.parent,
                                             prefix='.'+key+'.',delete=False) as stream:
                stream.write(canonical(record))
                temporary=stream.name
            # Atomic publication: another worker never sees partially written
            # JSON. Simultaneous writers publish the same content/hash.
            os.replace(temporary,destination)
        with self.db:
            self.db.execute('INSERT INTO records(kind, object_id) VALUES (?, ?)', (kind, key))
            if kind == 'experiment':
                value = json.loads(canonical(record))
                self.db.execute('INSERT OR IGNORE INTO experiment_index VALUES (?,?,?,?)',
                                (key,value['model_id'],value['workload_id'],int(value['valid'])))
        return key

    def load(self, key):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid content hash')
        value = json.loads((self.root / 'objects' / (key + '.json')).read_text())
        if identity(value) != key:
            raise ValueError('Corrupt artifact')
        return value

    def records(self, kind):
        return [self.load(row[0]) for row in self.db.execute(
            'SELECT object_id FROM records WHERE kind=? ORDER BY id', (kind,))]

    def close(self):
        self.db.close()
