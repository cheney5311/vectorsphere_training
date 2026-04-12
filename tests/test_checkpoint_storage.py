import os
import tempfile
from backend.utils.checkpoint_storage import S3Adapter


def test_s3adapter_mock_upload_download_list_and_presign(tmp_path):
    os.environ['MOCK_OBJECT_STORAGE'] = '1'
    # ensure mock root is isolated
    os.environ['MOCK_STORAGE_ROOT'] = str(tmp_path / 'mock_storage')

    adapter = S3Adapter(bucket='test-bucket')
    # create a temp file
    src = tmp_path / 'file.bin'
    src.write_bytes(b'hello-checkpoint')

    object_name = 'ckpt/test1.bin'
    # upload
    ok = adapter.upload(object_name, str(src))
    assert ok

    # list versions (mock should return the file)
    versions = adapter.list_versions('ckpt')
    assert any(v['key'].endswith('test1.bin') for v in versions)

    # download to new path
    dst = tmp_path / 'downloaded.bin'
    ok2 = adapter.download(object_name, str(dst))
    assert ok2
    assert dst.read_bytes() == b'hello-checkpoint'

    # presigned URL (mock returns file://)
    url = adapter.generate_presigned_url(object_name)
    assert url.startswith('file://')

    # cleanup env
    del os.environ['MOCK_OBJECT_STORAGE']
    del os.environ['MOCK_STORAGE_ROOT']
