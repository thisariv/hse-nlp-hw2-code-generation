"""Read-only one-step watch; repeated orchestrator calls never launch/cancel anything."""
import json
from pathlib import Path
import subprocess
import sys
from variants_common import *


def main():
    interactive=WORK/'results/kaggle-runs/nlp-hw02-variants/interactive'
    if (interactive/'variants/summary.json').is_file():
        summary=read(interactive/'variants/summary.json')
        if len(summary.get('variants',{}))==5:
            # Check the latest server code, including user-created versions.
            import variants_api as api
            version,proof=api.latest(api.connect())
            write(WORK/'checks/final-variants-download.json',dict(ref=KERNEL,version=version,
                output=str(interactive.relative_to(WORK)),downloaded_at_utc=api.now(),interactive=True,**proof))
            print(json.dumps(dict(ready=True,interactive=True,version=version)))
            return
    import variants_api as api
    api.observe(api.connect())

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps(dict(error_type=type(e).__name__,http=getattr(getattr(e,'response',None),'status_code',None),
            reason=str(e) if isinstance(e,AssertionError) else 'No accessible matching server version; no push attempted')))
        raise SystemExit(1)
