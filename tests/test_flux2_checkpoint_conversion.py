import torch
import torch.nn as nn

from model.backbone.flux2.checkpoint import (
    audit_mot_state_dict,
    convert_imagewam_flux2_checkpoint_payload,
    require_exact_imagewam_coverage,
)


def test_imagewam_top_level_proprio_key_is_renamed():
    payload, report = convert_imagewam_flux2_checkpoint_payload(
        {
            "mot": {"weight": torch.ones(2, 2)},
            "proprio_encoder": {"weight": torch.ones(3, 2)},
            "step": 7,
        }
    )
    assert "proprio_encoder" not in payload
    assert "state_encoder" in payload
    assert payload["backbone_name"] == "flux2"
    assert payload["model_variant"] == "mot"
    assert report["renamed_top_level"] == {"proprio_encoder": "state_encoder"}


def test_imagewam_mot_coverage_must_be_exact():
    module = nn.Linear(2, 3)
    report = audit_mot_state_dict(module, module.state_dict())
    require_exact_imagewam_coverage(report)
    assert report["numel_coverage"] == 1.0

    bad = audit_mot_state_dict(module, {"weight": torch.ones(4, 2)})
    try:
        require_exact_imagewam_coverage(bad)
    except RuntimeError as exc:
        assert "not exact" in str(exc)
    else:
        raise AssertionError("Shape/key mismatch must fail exact ImageWAM coverage.")
