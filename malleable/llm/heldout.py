"""Persistent, atomic consumption of declared held-out evaluation designs."""
import datetime
import json
from pathlib import Path
import sqlite3
from ..records import identity

DEFAULT_DESIGN='frozen-release-single-evaluation-v1'
LEDGER_PATH=Path(__file__).resolve().parents[2]/'docs/evidence/controller-quality-continuation-20261006/consumed-ledger.json'

def registry_path():
    return Path(__file__).resolve().parents[2]/'build/zephyrus-jobs/heldout-registry.sqlite3'

def evaluation_key(record,design=DEFAULT_DESIGN):
    configuration=record.get('configuration',record)
    if not isinstance(configuration,dict) or not isinstance(record.get('suite_hashes'),dict):
        raise ValueError('complete held-out configuration and suite identity required')
    fields={k:record.get(k) for k in ('base_model_id','tokenizer_id','variant_id','suite_file_hash')}
    fields.update(configuration_id=configuration.get('configuration_id'),split='held-out',
        split_hash=record.get('suite_hashes',{}).get('held-out'),evaluation_design=design)
    if any(not isinstance(v,str) or not v for v in fields.values()):
        raise ValueError('complete held-out model/tokenizer/variant/configuration/suite/design identity required')
    return fields

def durable_consumption(record,design=DEFAULT_DESIGN):
    """Return a published consumption tombstone, including lost-workspace claims."""
    if not LEDGER_PATH.is_file():
        return None
    key=evaluation_key(record,design)
    ledger=json.loads(LEDGER_PATH.read_text())
    for item in ledger.get('exact_evaluations',[]):
        if item.get('key')==key:
            return item
    for suite in ledger.get('suite_tombstones',[]):
        if (suite.get('evaluation_design')==design
            and suite.get('suite_file_hash')==key['suite_file_hash']
            and suite.get('heldout_split_hash')==key['split_hash']
            and suite.get('base_model_id')==key['base_model_id']
            and suite.get('tokenizer_id')==key['tokenizer_id']
            and suite.get('variant_id')==key['variant_id']
            and key['configuration_id'] in suite.get('configuration_ids',[])):
            return suite
    return None

def consume(record,design=DEFAULT_DESIGN,evidence=None):
    """INSERT is the claim; even failed computations permanently consume it."""
    key=evaluation_key(record,design); digest=identity(key)
    if durable_consumption(record,design) is not None:
        raise ValueError('held-out was already claimed in the published durable ledger; a separately documented evaluation design is required')
    path=registry_path(); path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'key':key,'evaluation_id':digest,
        'claimed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'freeze_id':record.get('freeze_id'),'evidence':evidence,'status':'consumed'}
    with sqlite3.connect(path,timeout=30) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS evaluations (evaluation_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        try: db.execute('INSERT INTO evaluations VALUES (?,?)',(digest,json.dumps(payload,sort_keys=True)))
        except sqlite3.IntegrityError:
            raise ValueError('held-out was already claimed; a separately documented evaluation design is required') from None
    return payload

def register_completed(record,result_path,result_sha256):
    """Import existing evidence without inference; idempotent only for identical evidence."""
    from .models import digest
    if record.get('split')!='held-out' or digest(result_path)!=result_sha256:
        raise ValueError('verified completed held-out result required for registry import')
    evidence={'result_sha256':result_sha256,'record_id':record.get('record_id')}
    try: return consume(record,evidence=evidence)
    except ValueError as error:
        key=identity(evaluation_key(record))
        with sqlite3.connect(registry_path()) as db:
            row=db.execute('SELECT payload FROM evaluations WHERE evaluation_id=?',(key,)).fetchone()
        if row and json.loads(row[0]).get('evidence')==evidence: return json.loads(row[0])
        raise error
