from backend.services.gpu_resource_manager import detect_gpus, get_gpu_metrics, get_gpu_summary


def test_gpu_detection_runs():
    gpus = detect_gpus()
    assert isinstance(gpus, list)

    metrics = get_gpu_metrics()
    assert isinstance(metrics, list)

    summary = get_gpu_summary()
    assert 'gpu_count' in summary
    assert 'gpus' in summary

