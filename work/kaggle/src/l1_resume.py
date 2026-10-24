"""Select the newest verified full main state, independently for each model."""
import json, shutil
from pathlib import Path
from kaggle_io import safe_name, sha, write

def select_resume(candidates,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rejected=[];selected={}
    for name in ('C','D'):
        choices=[]
        for source in candidates:
            root=Path(source)/f'model-{name}'
            if not (root/'latest.json').exists():continue
            try:
                raw=(root/'latest.json').read_bytes();rec=json.loads(raw)
                path=root/safe_name(rec['file'])
                assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
                assert rec['identity']['model']==name
                choices.append((rec['updates'],root,rec,raw))
            except (OSError,ValueError,KeyError,AssertionError) as error:
                rejected.append({'model':name,'source':str(source),'error_type':type(error).__name__})
        if not choices:raise ValueError('No verified full resume state for '+name)
        _,source,rec,raw=max(choices,key=lambda x:x[0]);target=destination/f'model-{name}'
        (target/rec['file']).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/rec['file'],target/rec['file']);(target/'latest.json').write_bytes(raw)
        assert sha(target/rec['file'])==rec['sha256']
        selected[name]={'updates':rec['updates'],'cursor':rec['cursor'],'source':str(source)}
    write(destination/'selection.json',{'selected':selected,'rejected':rejected})
    return destination
