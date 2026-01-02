from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
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
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        # pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
 
        bert_dur = self.bert(input_ids)

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
        
        # en = torch.repeat_interleave(d.transpose(-1,-2), pred_dur, dim=2)
        input_tensor = d.transpose(-1, -2)

        boundaries = torch.cumsum(pred_dur, dim=0)
        values = torch.arange(boundaries[-1], device=pred_dur.device)
        expanded_indices = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
        en = torch.index_select(input_tensor, 2, expanded_indices)

        print(f"\n\n\n\t{boundaries.shape=} {values.shape=} en: {en.shape=} {expanded_indices.shape=}\n\n\n")
        en, _ = self.predictor.shared(en.transpose(-1, -2))
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids)

        # The original line was:
        # asr = torch.repeat_interleave(t_en, pred_dur, dim=2)
        asr = torch.index_select(t_en, 2, expanded_indices)

        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        # logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        style: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, style, speed)
        return waveform, duration
