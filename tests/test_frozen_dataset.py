import json
from hashlib import sha256

import pytest

from research_engine.analysis.frozen_dataset import research_inputs


def test_frozen_inputs_ignore_new_files_and_duplicate_archive_copy_but_reject_mutation(tmp_path):
    archive,local=tmp_path/'archive',tmp_path/'local'
    sources=[]
    for origin,root,day in [('archive',archive,'2026-09-15'),('local',local,'2026-09-16')]:
        for file in ['features/'+day+'.parquet','normalized/'+day+'/events.parquet']:
            path=root/file
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'fixed content')
            sources.append(dict(origin=origin,file=file,sha256=sha256(path.read_bytes()).hexdigest()))
    (archive/'features/2026-09-16.parquet').write_bytes(b'other copy')
    (archive/'features/2026-09-17.parquet').write_bytes(b'later file')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(original_dates=['2026-09-15'],added_dates=['2026-09-16'],sources=sources)))
    inputs,_=research_inputs(archive,local,manifest)
    assert list(inputs)==['2026-09-15','2026-09-16']
    assert inputs['2026-09-16'][0]=='local'
    (local/'features/2026-09-16.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        research_inputs(archive,local,manifest)
