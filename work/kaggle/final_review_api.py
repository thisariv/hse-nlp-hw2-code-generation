"""Private final benchmark API. Environment authorization only; credential files never read.

Use the interpreter in the installed Kaggle CLI shebang. Never prints credentials,
signed URLs, HTTP bodies, or exception text. Does not delete remote resources.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = 'thisarivson/nlp-hw02-final-inference'


def environment_api():
    if not (os.environ.get('KAGGLE_API_TOKEN') or
            (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))):
        raise RuntimeError('environment_authorization_unavailable')
    # kaggle.__init__ authenticates immediately. Load the official submodule without
    # executing that initializer; configure the SDK explicitly from the environment.
    spec = importlib.util.find_spec('kaggle')
    if spec is None:
        raise RuntimeError('kaggle_sdk_unavailable')
    package = types.ModuleType('kaggle')
    package.__path__ = list(spec.submodule_search_locations)
    package.__version__ = importlib.metadata.version('kaggle')
    sys.modules['kaggle'] = package
    os.environ['KAGGLE_CONFIG_DIR'] = tempfile.mkdtemp(prefix='final-api-no-credentials-')
    os.environ.pop('VERBOSE', None)
    os.environ.pop('VERBOSE_OUTPUT', None)
    from kaggle.api.kaggle_api_extended import KaggleApi
    import kagglesdk.kaggle_http_client as http
    http.get_access_token_from_env = lambda: (os.environ.get('KAGGLE_API_TOKEN'), 'environment')
    api = KaggleApi()
    api.config_values = api.read_config_environment({})
    if os.environ.get('KAGGLE_API_TOKEN'):
        api.config_values[api.CONFIG_NAME_TOKEN] = os.environ['KAGGLE_API_TOKEN']
    api._authenticated = True
    return api


def private_inputs(api):
    required = {'thisarivson/nlp-hw02-l2-live', 'thisarivson/nlp-hw02-26b-inputs'}
    found = {}
    for page in range(1, 100):
        rows = api.dataset_list(mine=True, page=page) or []
        for row in rows:
            if row.ref in required:
                assert row.is_private, 'Input must be private'
                found[row.ref] = row.current_version_number
        if required == found.keys() or not rows:
            break
    assert required == found.keys(), 'Required private datasets unavailable'
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['probe', 'push', 'status', 'output'])
    parser.add_argument('--version', type=int)
    parser.add_argument('--iteration', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    api = environment_api()
    from kaggle_versioned import pin_client, output
    if args.command == 'probe':
        print(json.dumps({'authorized': True, 'private_inputs': private_inputs(api),
                          'credential_files_read': False}))
        return
    if args.command == 'push':
        inputs = private_inputs(api)
        bundle = ROOT/'bundles/final_inference'
        metadata = json.loads((bundle/'kernel-metadata.json').read_text())
        assert metadata['id'] == REF and metadata['is_private'] is True
        assert metadata['enable_gpu'] is True and metadata['enable_internet'] is False
        assert metadata['machine_shape'] == 'NvidiaTeslaT4'
        assert json.loads((bundle/'session-limits.json').read_text())['session_timeout_seconds'] == 900
        registry_path = ROOT/'runs.json'
        registry = json.loads(registry_path.read_text())
        review = registry['final_inference'].setdefault('independent_review', {})
        attempts = review.setdefault('API_push_attempts', [])
        # Record before sending: an ambiguous network response must not cause an automatic repush.
        assert len(attempts) < 2 and len(attempts)+1 == args.iteration
        attempts.append({'iteration': args.iteration, 'started_at': datetime.now(timezone.utc).isoformat(),
                         'status': 'request_started', 'private_inputs': inputs})
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n')
        response = api.kernels_push(str(bundle), timeout='900', acc='NvidiaTeslaT4')
        assert not response.error and response.versionNumber, 'Push response rejected'
        assert not response.invalidDatasetSources and not response.invalidKernelSources
        attempts[-1].update(status='submitted', version=response.versionNumber)
        review.update(status='submitted', version=response.versionNumber)
        registry['final_inference'].update(pushed=True)
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'ref': REF, 'version': response.versionNumber, 'iteration': args.iteration,
                          'private': True, 'timeout_seconds': 900}))
        return
    assert args.version and args.version > 0, 'Specify the version returned by push'
    ref = REF+'/'+str(args.version)
    if args.command == 'status':
        pin_client(api, ref)
        print(json.dumps({'ref': ref, 'status': api.kernels_status(ref).status.name}))
    else:
        target = ROOT.parent/'results/kaggle-runs/nlp-hw02-final-inference'/str(args.version)
        output(api, ref, target)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # No exception text: HTTP errors may contain signed URLs or headers.
        safe_reason = str(error) if isinstance(error, RuntimeError) and str(error) in {
            'environment_authorization_unavailable', 'kaggle_sdk_unavailable'} else None
        print(json.dumps({'status': 'blocked', 'reason': safe_reason, 'error_type': type(error).__name__,
                          'http_status': getattr(getattr(error, 'response', None), 'status_code', None),
                          'credential_files_read': False}))
        sys.exit(3)
