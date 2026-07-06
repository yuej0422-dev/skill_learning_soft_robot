from __future__ import annotations


def torch_device_report() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"torch_importable": False, "error": str(exc)}
    report = {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(props.total_memory / (1024**3), 3),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return report

