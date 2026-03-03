"""Core Kokoro model with configurable decoder backends.

This module defines `KModel`, which performs text/phoneme conditioning and
dispatches vocoder synthesis to one of:
- `decoder_type="pt_vocos"` (default)
- `decoder_type="tf_vocos"`
- `decoder_type="istft"`

Config examples:
1) Default PT Vocos:
   {
     "decoder_type": "pt_vocos",
     "vocos": {"checkpoint_path": "/path/to/vocos_last.pt"}
   }

2) TensorFlow Vocos:
   {
     "decoder_type": "tf_vocos",
     "vocos": {"checkpoint_path": "/path/to/vocos_last.pt"}
   }

3) Legacy iSTFT:
   {
     "decoder_type": "istft"
   }

4) PT Vocos streaming decoder:
   {
     "decoder_type": "pt_vocos",
     "vocos": {
       "checkpoint_path": "/path/to/vocos_last.pt",
       "streaming": true,
       "chunk_size_ms": 300
     }
   }

Streaming note:
- Chunked parity semantics in this repo are:
  1) run conditioner once on full vocoder features,
  2) chunk only backbone + ISTFT head.
- PT streaming decoder uses streaming-vocos module contexts.
- TF streaming decoder mirrors the same chunk boundaries/conditioning strategy.

Quick smoke test:
`uv run python -c "from kokoro import KModel; m=KModel(config='/path/config.json', model='/path/kokoro.pth'); print(type(m.decoder).__name__)"`
"""

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from .vocos_decoder import (
    PTVocosDecoder,
    StreamingPTVocosDecoder,
    StreamingTFVocosDecoder,
    TFVocosDecoder,
)
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout = 0 #dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder_type = str(config.get('decoder_type', config.get('decoder', 'pt_vocos'))).lower()
        vocos_cfg = config.get('vocos', config.get('vocos_decoder', {}))
        if self.decoder_type == 'istft':
            self.decoder = Decoder(
                dim_in=config['hidden_dim'], style_dim=config['style_dim'],
                dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
            )
        elif self.decoder_type == 'pt_vocos':
            use_streaming = bool(vocos_cfg.get('streaming', False))
            decoder_cls = StreamingPTVocosDecoder if use_streaming else PTVocosDecoder
            decoder_kwargs = dict(
                dim_in=config['hidden_dim'],
                style_dim=config['style_dim'],
                model_input_channels=int(vocos_cfg.get('model_input_channels', 192)),
                backbone_dim=int(vocos_cfg.get('backbone_dim', 384)),
                backbone_intermediate_dim=int(vocos_cfg.get('backbone_intermediate_dim', 1152)),
                backbone_layers=int(vocos_cfg.get('backbone_layers', 8)),
                n_fft=int(vocos_cfg.get('n_fft', 1200)),
                hop_length=int(vocos_cfg.get('hop_length', 300)),
                checkpoint_path=vocos_cfg.get('checkpoint_path'),
                vocos_impl=str(vocos_cfg.get('vocos_impl', 'auto')),
                streaming_vocos_repo=vocos_cfg.get('streaming_vocos_repo', 'third_party/vocos_streaming'),
                backbone_causal=bool(vocos_cfg.get('backbone_causal', True)),
                backbone_pad_mode=str(vocos_cfg.get('backbone_pad_mode', 'constant')),
                backbone_norm=str(vocos_cfg.get('backbone_norm', 'weight_norm')),
            )
            if not use_streaming:
                decoder_kwargs["padding"] = str(vocos_cfg.get('padding', 'same'))
            if use_streaming:
                decoder_kwargs.update(
                    sample_rate=int(vocos_cfg.get('sample_rate', 24000)),
                    chunk_size_ms=int(vocos_cfg.get('chunk_size_ms', 300)),
                )
            self.decoder = decoder_cls(**decoder_kwargs)
        elif self.decoder_type == 'tf_vocos':
            use_streaming = bool(vocos_cfg.get('streaming', False))
            decoder_cls = StreamingTFVocosDecoder if use_streaming else TFVocosDecoder
            decoder_kwargs = dict(
                dim_in=config['hidden_dim'],
                style_dim=config['style_dim'],
                model_input_channels=int(vocos_cfg.get('model_input_channels', 192)),
                backbone_dim=int(vocos_cfg.get('backbone_dim', 384)),
                backbone_intermediate_dim=int(vocos_cfg.get('backbone_intermediate_dim', 1152)),
                backbone_layers=int(vocos_cfg.get('backbone_layers', 8)),
                n_fft=int(vocos_cfg.get('n_fft', 1200)),
                hop_length=int(vocos_cfg.get('hop_length', 300)),
                checkpoint_path=vocos_cfg.get('checkpoint_path'),
                vocos_impl=str(vocos_cfg.get('vocos_impl', 'auto')),
                streaming_vocos_repo=vocos_cfg.get('streaming_vocos_repo', 'third_party/vocos_streaming'),
                backbone_causal=bool(vocos_cfg.get('backbone_causal', True)),
                backbone_pad_mode=str(vocos_cfg.get('backbone_pad_mode', 'constant')),
                backbone_norm=str(vocos_cfg.get('backbone_norm', 'weight_norm')),
            )
            if not use_streaming:
                decoder_kwargs["padding"] = str(vocos_cfg.get('padding', 'same'))
            if use_streaming:
                decoder_kwargs.update(
                    sample_rate=int(vocos_cfg.get('sample_rate', 24000)),
                    chunk_size_ms=int(vocos_cfg.get('chunk_size_ms', 300)),
                )
            self.decoder = decoder_cls(**decoder_kwargs)
        else:
            raise ValueError(f"Unknown decoder_type='{self.decoder_type}'. Supported: istft, pt_vocos, tf_vocos.")
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            if not hasattr(self, key):
                logger.debug(f"Skipping unknown checkpoint key: {key}")
                continue
            self._load_module_state_dict(getattr(self, key), state_dict, module_name=key)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        vocoder_io: Optional['KModel.VocoderIO'] = None
        pred_dur: Optional[torch.LongTensor] = None

    @dataclass
    class VocoderIO:
        # Input tensors consumed by all decoder variants.
        asr: torch.FloatTensor
        f0: torch.FloatTensor
        noise: torch.FloatTensor
        style: torch.FloatTensor

    @staticmethod
    def _strip_parallel_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fixed: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "_orig_mod."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            fixed[new_key] = value
        return fixed

    def _load_module_state_dict(self, module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], module_name: str) -> None:
        state_dict = self._strip_parallel_prefixes(state_dict)
        try:
            module.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            pass

        module_state = module.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            expected = module_state.get(key)
            if expected is not None and tuple(expected.shape) == tuple(value.shape):
                filtered[key] = value
            else:
                skipped.append(key)

        result = module.load_state_dict(filtered, strict=False)
        skipped_count = len(skipped)
        if skipped_count > 0 or result.missing_keys or result.unexpected_keys:
            logger.debug(
                f"Partial load for {module_name}: loaded={len(filtered)} "
                f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)} skipped={skipped_count}"
            )

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_vocoder_io: bool = False,
    ) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, 'KModel.VocoderIO']]:
        bert_dur = self.bert(input_ids)
        bert_dur = bert_dur.last_hidden_state

        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s)

        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).squeeze()

        # Alignment/upsampling of phoneme embeddings to match audio frame rate.
        # This replaces torch.repeat_interleave with an equivalent operation
        # using torch.index_select for broader compatibility.
        input_tensor = d.transpose(-1, -2)

        boundaries = torch.cumsum(pred_dur, dim=0)
        values = torch.arange(boundaries[-1], device=pred_dur.device)
        expanded_indices = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
        en = torch.index_select(input_tensor, 2, expanded_indices)

        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids)

        # The original line was:
        # asr = torch.repeat_interleave(t_en, pred_dur, dim=2)
        asr = torch.index_select(t_en, 2, expanded_indices)

        vocoder_io = None
        if return_vocoder_io:
            vocoder_io = self.VocoderIO(
                asr=asr.squeeze(0).detach().cpu(),
                f0=F0_pred.squeeze(0).detach().cpu(),
                noise=N_pred.squeeze(0).detach().cpu(),
                style=ref_s[:, :128].squeeze(0).detach().cpu(),
            )
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        if return_vocoder_io:
            return audio, vocoder_io
        return audio

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
        return_vocoder_io: bool = False,
    ) -> Union['KModel.Output', torch.FloatTensor, tuple[torch.FloatTensor, 'KModel.VocoderIO']]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        vocoder_io = None
        output = self.forward_with_tokens(
            input_ids,
            ref_s,
            speed,
            return_vocoder_io=return_vocoder_io,
        )
        if return_vocoder_io:
            audio, vocoder_io = output
        else:
            audio = output
        audio = audio.squeeze().cpu()
        # logger.debug(f"pred_dur: {pred_dur}")
        if return_output:
            return self.Output(audio=audio, vocoder_io=vocoder_io)
        if return_vocoder_io:
            return audio, vocoder_io
        return audio

    @torch.no_grad()
    def stream_decode_vocoder_io(
        self,
        asr: torch.Tensor,
        f0: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
        is_last: bool = False,
    ):
        """Stream waveform chunks from vocoder inputs when decoder supports it."""
        if not hasattr(self.decoder, "streaming_decode"):
            raise RuntimeError("Configured decoder does not support streaming_decode")
        asr = asr.to(self.device)
        f0 = f0.to(self.device)
        noise = noise.to(self.device)
        style = style.to(self.device)
        yield from self.decoder.streaming_decode(asr, f0, noise, style, is_last=is_last)

    @torch.no_grad()
    def reset_decoder_stream(self) -> None:
        if hasattr(self.decoder, "reset"):
            self.decoder.reset()

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform
