import os
import tempfile
import shutil
from backend.utils.checkpoint_manager import get_checkpoint_manager
from backend.utils.checkpoint_storage import S3Adapter


def test_checkpoint_save_and_upload(tmp_path, monkeypatch):
    # 使用临时目录作为 checkpoint dir
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    manager = get_checkpoint_manager(str(cp_dir))

    # 准备一个最小的 model-like 对象
    class DummyModel:
        def state_dict(self):
            return {'w': 1}

    model = DummyModel()

    # 设置本地模拟 bucket
    bucket = 'test-bucket'
    os.environ['CHECKPOINT_BUCKET'] = bucket

    # 清理模拟存储目录
    mock_bucket_dir = os.path.join('/tmp', 's3mock', bucket)
    if os.path.exists(mock_bucket_dir):
        shutil.rmtree(mock_bucket_dir)

    # 保存 checkpoint（会尝试上传到本地模拟 bucket）
    path = manager.save_checkpoint(model, epoch=1, step=1)
    assert os.path.exists(path)

    # 元数据文件应存在并包含 bucket/object_name
    meta_path = path + '.meta.json'
    assert os.path.exists(meta_path)
    with open(meta_path, 'r') as mf:
        meta = __import__('json').load(mf)
    assert meta.get('bucket') == bucket
    assert 'object_name' in meta

    # 使用 S3Adapter 下载验证
    adapter = S3Adapter(bucket=bucket)
    object_name = meta.get('object_name')
    tmp_download = str(tmp_path / 'dl.ckpt')
    ok = adapter.download(object_name, tmp_download)
    assert ok and os.path.exists(tmp_download)

    # cleanup
    del os.environ['CHECKPOINT_BUCKET']
