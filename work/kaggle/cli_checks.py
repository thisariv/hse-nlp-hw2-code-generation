#!/usr/bin/env python3
"""Offline installed-CLI request checks; run with the Kaggle launcher's Python."""
import contextlib,importlib.metadata,io,json,os,pathlib,tempfile,types
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parent

def main():
    with tempfile.TemporaryDirectory(prefix='nlp-kaggle-cli-') as tmp:
        tmp=pathlib.Path(tmp)
        # Import-time Kaggle authentication must not inspect the real credential directory.
        os.environ['KAGGLE_CONFIG_DIR']=str(tmp/'empty-config')
        os.environ['KAGGLE_API_TOKEN']='offline-test-placeholder'
        with contextlib.redirect_stdout(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        import requests
        from kaggle_versioned import pin_client,output
        from kaggle_io import sha,write
        assert importlib.metadata.version('kaggle')=='2.2.4'
        api=KaggleApi();calls=[]
        class Target:
            def save_kernel(self,r):
                calls.append({'method':'save','shape':r.machine_shape,'gpu':r.enable_gpu,'tpu':r.enable_tpu,
                              'private':r.is_private,'internet':r.enable_internet,'timeout':r.session_timeout_seconds})
                return types.SimpleNamespace()
            def get_kernel_session_status(self,r):
                calls.append({'method':'status','slug':r.kernel_slug,'version':r.version_label})
                return types.SimpleNamespace(status=types.SimpleNamespace(name='RUNNING'))
            def get_kernel(self,r):
                calls.append({'method':'pull','slug':r.kernel_slug,'version':r.version_label})
                return types.SimpleNamespace()
            def list_kernel_session_output(self,r):
                calls.append({'method':'output','slug':r.kernel_slug,'version':r.version_label,'page':r.page_token})
                name='second.json' if r.page_token else 'nested/first.json'
                return types.SimpleNamespace(files=[types.SimpleNamespace(file_name=name,url='https://offline.invalid/'+name)],
                                             log='mock log',next_page_token='' if r.page_token else 'page2')
        @contextlib.contextmanager
        def fake_client():
            yield types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=Target()))
        api.build_kaggle_client=fake_client
        for kind in ('s1','26c','s2'):
            folder=tmp/kind;folder.mkdir()
            meta=json.loads((ROOT/'bundles'/kind/'kernel-metadata.json').read_text())
            (folder/meta['code_file']).write_text('{"cells":[],"nbformat":4,"nbformat_minor":5,"metadata":{}}')
            write(folder/'kernel-metadata.json',meta)
            api.kernels_push(str(folder),timeout='13200' if kind=='s2' else '43200',acc='NvidiaTeslaT4' if kind in ('s1','s2') else None)
        assert calls[0]=={'method':'save','shape':'NvidiaTeslaT4','gpu':True,'tpu':False,'private':True,'internet':True,'timeout':43200}
        assert calls[1]=={'method':'save','shape':'','gpu':False,'tpu':False,'private':True,'internet':True,'timeout':43200}
        assert calls[2]==dict(calls[0],timeout=13200)
        pin_client(api,'offline/example-notebook/7')
        api.kernels_status('offline/example-notebook/7')
        with api.build_kaggle_client() as client:
            r=ApiGetKernelRequest();r.user_name='offline';r.kernel_slug='example-notebook/7'
            client.kernels.kernels_api_client.get_kernel(r)
        class Stream:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def raise_for_status(self):pass
            def iter_content(self,*a):yield b'{"';yield b'checked":true}'
        # Restore unpinned fake client before testing another immutable version.
        api.build_kaggle_client=fake_client
        with patch.object(requests,'get',return_value=Stream()):
            output(api,'offline/example-notebook/8',tmp/'output')
        assert json.loads((tmp/'output/nested/first.json').read_text())=={'checked':True}
        assert (tmp/'output/second.json').exists()
        assert calls[3:5]==[{'method':'status','slug':'example-notebook','version':'v7'},
                           {'method':'pull','slug':'example-notebook','version':'v7'}]
        assert [c['version'] for c in calls[5:]]==['v8','v8'] and calls[-1]['page']=='page2'
        source=pathlib.Path(__import__('inspect').getfile(KaggleApi))
        write(ROOT.parent/'checks/kaggle-cli-224-metadata-checks.json',{
            'CLI':'2.2.4','metadata_field':'machine_shape','T4x2':'NvidiaTeslaT4',
            'GPU_request_serialization_mock_checked':True,'CPU_request_serialization_mock_checked':True,
            'S2_session_timeout_request_seconds':13200,'S2_timeout_serialization_mock_checked':True,
            'versioned_status_pull_paged_download_mock_checked':True,'version_label_format':'vN',
            'API_source_sha256':sha(source),'network_save_calls':0,'network_calls_in_this_check':0,
            'official_docs':'https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md',
            'official_version_fix':'https://github.com/Kaggle/kaggle-cli/pull/1201'})
        print('PASSED CLI2.2.4 private/T4x2/CPU requests and pinned vN status/pull/paged download; no network')

if __name__=='__main__':main()
