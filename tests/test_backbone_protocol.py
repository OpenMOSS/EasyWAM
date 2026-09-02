import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import types

from model.backbone.loader import normalize_backbone_config
from model.component.mot import MoT
from model.easywam_mot import EasyWAMMoT


class _MainBlock(nn.Module):
    def prepare_mixed_attention(self, x, embedding, adaln_lora, freqs, token_to_timestep):
        del embedding, adaln_lora, freqs, token_to_timestep
        return x, x, x, x

    def finish_mixed_attention(self, mixed, residual, context, context_mask):
        del context, context_mask
        return residual + mixed


class _MainExpert(nn.Module):
    num_heads = 1
    attn_head_dim = 2
    attention_backend = "sdpa"
    use_gradient_checkpointing = False

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_MainBlock()])


def test_backbone_normalizer_accepts_flux2():
    assert normalize_backbone_config({"name": "FLUX2"})["name"] == "flux2"


def test_main_protocol_keeps_homogeneous_block_execution():
    mot = MoT({"video": _MainExpert(), "action": _MainExpert()})
    video = torch.tensor([[[1.0, 0.0]]])
    action = torch.tensor([[[0.0, 1.0]]])
    joined = torch.cat([video, action], dim=1)
    expected_attention = F.scaled_dot_product_attention(
        joined.unsqueeze(1), joined.unsqueeze(1), joined.unsqueeze(1)
    ).squeeze(1)
    output = mot(
        embeds_all={"video": video, "action": action},
        attention_mask=torch.ones(2, 2, dtype=torch.bool),
        freqs_all={"video": None, "action": None},
        context_all={"video": None, "action": None},
        t_mod_all={
            "video": {"embedding": None, "adaln_lora": None},
            "action": {"embedding": None, "adaln_lora": None},
        },
    )
    assert mot.block_protocol == "main"
    torch.testing.assert_close(output["video"], video + expected_attention[:, :1])
    torch.testing.assert_close(output["action"], action + expected_attention[:, 1:])


def test_flux2_action_mask_uses_text_reference_and_action_not_noisy_target():
    mask = EasyWAMMoT._build_mot_attention_mask_flux2(
        None,
        batch_size=1,
        txt_len=2,
        cond_len=3,
        target_len=4,
        action_len=2,
        device=torch.device("cpu"),
        text_attention_mask=torch.tensor([[True, False]]),
    )["double_joint"][0]
    action_rows = mask[9:11]
    assert action_rows[:, 0].all()
    assert not action_rows[:, 1].any()
    assert action_rows[:, 2:5].all()
    assert not action_rows[:, 5:9].any()
    assert action_rows[:, 9:11].all()


def test_flux2_protocol_runs_double_then_single_stage_without_official_weights():
    flux2_package = types.ModuleType("flux2")
    flux2_model = types.ModuleType("flux2.model")

    class QKNorm(nn.Module):
        def __init__(self, head_dim):
            super().__init__()
            self.head_dim = head_dim

        def forward(self, q, k, v):
            del v
            return q, k

    class SiLUActivation(nn.Module):
        def forward(self, x):
            value, gate = x.chunk(2, dim=-1)
            return value * F.silu(gate)

    class MLPEmbedder(nn.Module):
        def __init__(self, in_dim, hidden_dim, disable_bias=True):
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)

        def forward(self, x):
            return self.proj(x)

    class Modulation(nn.Module):
        def __init__(self, hidden_dim, double, disable_bias=True):
            super().__init__()
            del disable_bias
            self.hidden_dim = hidden_dim
            self.double = double

        def forward(self, vec):
            zero = torch.zeros_like(vec)[:, None]
            one = torch.ones_like(vec)[:, None]
            mod = (zero, zero, one)
            return (mod, mod) if self.double else (mod, None)

    flux2_model.QKNorm = QKNorm
    flux2_model.SiLUActivation = SiLUActivation
    flux2_model.MLPEmbedder = MLPEmbedder
    flux2_model.Modulation = Modulation
    flux2_model.timestep_embedding = lambda timestep, dim: timestep[:, None].expand(-1, dim)
    flux2_model.apply_rope = lambda q, k, pe: (q, k)
    flux2_package.model = flux2_model
    sys.modules["flux2"] = flux2_package
    sys.modules["flux2.model"] = flux2_model

    from model.component.action_dit_flux2 import ActionDiTFlux2

    action_expert = ActionDiTFlux2(
        action_dim=2,
        hidden_dim=8,
        num_heads=2,
        attn_head_dim=4,
        num_layers_double=1,
        num_layers_single=1,
        max_action_horizon=4,
    )

    class VideoDouble(nn.Module):
        def _prepare_qkv(self, img, txt, img_pe, txt_pe, img_mod, txt_mod):
            del img_pe, txt_pe, img_mod, txt_mod
            stream = torch.cat([txt, img], dim=1)
            qkv = stream.view(stream.shape[0], stream.shape[1], 2, 4).permute(0, 2, 1, 3)
            return qkv, qkv, qkv, None, txt.shape[1], None

        def _apply_residuals(self, img, txt, img_attn, txt_attn, mods):
            del mods
            return img + img_attn, txt + txt_attn

    class VideoSingle(nn.Module):
        def _qkv(self, x, modulation):
            del modulation
            qkv = x.view(x.shape[0], x.shape[1], 2, 4).permute(0, 2, 1, 3)
            return qkv, qkv, qkv, x, torch.ones_like(x[:, :1])

        def _out(self, residual, attention, mlp, gate):
            del mlp, gate
            return residual + attention

    class Transformer(nn.Module):
        @staticmethod
        def pe_embedder(ids):
            return ids[:, None]

    class VideoExpert(nn.Module):
        block_protocol = "flux2"
        num_heads = 2
        attn_head_dim = 4
        attention_backend = "sdpa"
        double_layers = 1
        single_layers = 1
        use_gradient_checkpointing = False

        def __init__(self):
            super().__init__()
            self.double_blocks = nn.ModuleList([VideoDouble()])
            self.single_blocks = nn.ModuleList([VideoSingle()])
            self.transformer = Transformer()

        @property
        def blocks(self):
            return list(self.double_blocks) + list(self.single_blocks)

    video_expert = VideoExpert()
    mot = MoT({"video": video_expert, "action": action_expert})
    action_pre = action_expert.pre_dit(
        torch.zeros(1, 1, 2), torch.full((1,), 0.5)
    )
    output = mot(
        embeds_all={
            "video": {"txt": torch.ones(1, 1, 8), "img": torch.ones(1, 1, 8)},
            "action": action_pre["tokens"],
        },
        attention_mask={
            "double_joint": torch.ones(1, 3, 3, dtype=torch.bool),
            "single": torch.ones(1, 3, 3, dtype=torch.bool),
        },
        freqs_all={
            "video": {"txt": torch.zeros(1, 1, 1, 4), "img": torch.zeros(1, 1, 1, 4)},
            "action": None,
        },
        context_all={"video": None, "action": {"ids": action_pre["ids"]}},
        t_mod_all={
            "video": {
                "double_img": None,
                "double_txt": None,
                "single": None,
            },
            "action": action_pre["t_mod"],
        },
    )
    assert output["video"]["txt"].shape == (1, 1, 8)
    assert output["video"]["img"].shape == (1, 1, 8)
    assert output["action"].shape == (1, 1, 8)
    assert "head.adaLN_modulation.1.weight" in action_expert.state_dict()
