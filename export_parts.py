import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pydoc import text
from matplotlib import style
from regex import F
import torch
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
import sounddevice as sd
import numpy as np
from torch.export import Dim
from onnxruntime.transformers.float16 import convert_float_to_float16

from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX
from kokoro.istftnet import export_compatible_ops
import matplotlib

# import ai_edge_torch

##########################################################################
OPSET_VERSION = 19
MAX_INPUT_LENGTH = 510
MAX_EXPANDED_LENGTH = 1024
AUDIO_SAMPLES_PER_FRAME = 600
MAX_AUDIO_LENGTH = MAX_EXPANDED_LENGTH * AUDIO_SAMPLES_PER_FRAME
DEFAULT_SAMPLE_TEXT = """
    The sky above the port was the color of television, tuned to a dead channel.
    """


def save_fp16_onnx(onnx_file):
    fp16_file = os.path.splitext(onnx_file)[0] + ".fp16.onnx"
    # Keep FP16 export as a plain dtype conversion path; do not run onnxsim here.
    fp16_model = convert_float_to_float16(
        onnx.load(onnx_file),
        keep_io_types=True,
    )
    fp16_model = gs.export_onnx(gs.import_onnx(fp16_model).toposort())
    onnx.save(fp16_model, fp16_file)
    onnx.checker.check_model(onnx.load(fp16_file))
    print(f"export {os.path.basename(fp16_file)} ok!")
    print("onnx fp16 check ok!")
    return fp16_file

def export_bert(model, input, output_dir):
    onnx_file = output_dir + "/" + "bert.onnx"

    (input_ids, text_mask) = input

    batch_size = Dim.STATIC #Dim("batch_size", min=1, max=32)
    input_len = Dim("seq_length", min=2, max=510)

    class model_bert(torch.nn.Module):
        def __init__(self, model):
            super(model_bert, self).__init__()
            self.bert = model.bert
            self.bert_encoder = model.bert_encoder

        def forward(self, input_ids, text_mask=text_mask):
            bert_dur = self.bert(input_ids, attention_mask=text_mask)
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            return d_en

    model_bert_instance = model_bert(model).eval()

    d_en = model_bert_instance(input_ids, text_mask)
    print(f'\nmodel_bert_instance input: {input_ids.shape=} bert output d_en: {d_en.shape=}\n')

    torch.onnx.export(
        model_bert_instance, 
        args =  (input_ids,), 
        f = onnx_file, 
        export_params = True, 
        verbose = False, 
        input_names = [ 'input_ids', 'text_mask' ], 
        output_names = [ 'd_en' ],
        opset_version = OPSET_VERSION, 
        # dynamic_shapes = { 'input_ids': {0: batch_size, 1: input_len} },
        do_constant_folding = True, 
        dynamo = True,
        external_data=False,
        # report = True,
    )

    # edge_model = ai_edge_torch.convert(model_bert_instance.eval(),
    #     sample_args =  (input_ids,), 
    #     strict_export = True, 
    #     dynamic_shapes = { 'input_ids': {1: input_len} },
    #     )


    print('export bert.onnx ok!')
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')
    save_fp16_onnx(onnx_file)
    return d_en

def export_duration_predictor(model, input, output_dir):
    onnx_file = output_dir + "/" + "duration_predictor.onnx"

    (input_ids, d_en, style, input_length, speed) = input

    batch_size = Dim.STATIC #Dim("batch_size", min=1, max=32)
    input_len = Dim("seq_length", min=2, max=510)
    feature_dim = Dim.STATIC #Dim("feature_dim", min=64, max=512)
    
    class model_duration_predictor(torch.nn.Module):
        def __init__(self, model):
            super(model_duration_predictor, self).__init__()
            self.predictor = model.predictor
            self.text_encoder = model.text_encoder

        def forward(self, input_ids, d_en, style, input_length, speed):
            input_length_i64 = input_length.to(torch.int64).reshape(())
            input_ids = input_ids[:, :input_length_i64]
            d_en = d_en[:, :, :input_length_i64]
            text_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.float32, device=input_ids.device)
            d = self.predictor.text_encoder(d_en, style[:, 128:], text_mask)
            x, _ = self.predictor.lstm(d)
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur_actual = torch.round(duration).clamp(min=1).reshape(-1)
            pred_dur = torch.nn.functional.pad(pred_dur_actual, (0, MAX_INPUT_LENGTH - pred_dur_actual.shape[0]))

            input_tensor = d.transpose(-1, -2)
            d_padded = torch.nn.functional.pad(input_tensor, (0, MAX_INPUT_LENGTH - input_tensor.shape[-1]))
            valid_token_count = input_length_i64.to(torch.int32)

            boundaries = torch.cumsum(pred_dur_actual, dim=0)
            print(f'\n\n\nboundaries: {boundaries.shape=} {boundaries=}\n\n\n')
            expanded_length = torch.clamp(boundaries[-1].to(torch.int32), max=MAX_EXPANDED_LENGTH)
            expanded_length_i64 = expanded_length.to(torch.int64)
            values = torch.arange(MAX_EXPANDED_LENGTH, device=pred_dur.device, dtype=torch.float32)
            expanded_indices = torch.sum(
                (boundaries.unsqueeze(1) <= values.unsqueeze(0)).to(torch.float32),
                dim=0,
            ).to(torch.int32)
            expanded_indices = torch.clamp(expanded_indices, max=valid_token_count - 1)
            print(f"\n\n\nvalues: {expanded_indices.shape=} {expanded_indices=}\n\n\n")
            actual_indices = expanded_indices[:expanded_length_i64]
            actual_en = torch.index_select(input_tensor, 2, actual_indices)
            actual_en, _ = self.predictor.shared(actual_en.transpose(-1, -2)) # run shared only on valid frames, then pad
            en = torch.nn.functional.pad(actual_en, (0, 0, 0, MAX_EXPANDED_LENGTH - actual_en.shape[1]))
            t_en = self.text_encoder(input_ids, text_mask)
            t_en = torch.nn.functional.pad(t_en, (0, MAX_INPUT_LENGTH - t_en.shape[-1]))

            return pred_dur, d_padded, expanded_indices, en, t_en, expanded_length

    with torch.no_grad():
        model_duration_instance = model_duration_predictor(model).eval()

        pred_dur, input_tensor, expanded_indices, en, t_en, expanded_length = model_duration_instance(input_ids, d_en, style, input_length, speed)
        print(f'\nmodel_duration_instance: {input_ids.shape=} {d_en.shape=} {pred_dur.shape=} {input_tensor.shape=} {en.shape=} {t_en.shape=} {expanded_indices=} {expanded_length=}\n\n')

        torch.onnx.export(
            model_duration_instance, 
            args =  (input_ids, d_en, style, input_length, speed), 
            f = onnx_file, 
            export_params = True, 
            verbose = False, 
            input_names = [ 'input_ids', 'd_en', 'style', 'input_length', 'speed' ], 
            output_names = [ 'pred_dur', 'd', 'expanded_indices', 'en', 't_en', 'expanded_length' ],
            opset_version = OPSET_VERSION, 
            do_constant_folding = True, 
            dynamo = False,
            external_data=False,
            # report = True,
        )


    # torch.onnx.export(
    #     model_duration_instance, 
    #     args =  (d_en, style, speed), 
    #     f = onnx_file, 
    #     export_params = True, 
    #     verbose = True, 
    #     input_names = [ 'd_en', 'style', 'speed' ], 
    #     output_names = [ 'pred_dur', 'd' ],
    #     opset_version = OPSET_VERSION, 
    #     dynamic_shapes = { 
    #         'd_en': {0: batch_size, 1: feature_dim, 2: input_len}, 
    #         'style': {0: batch_size}, 
    #         'speed': {0: batch_size}
    #     },
    #     do_constant_folding = True, 
    #     dynamo = True,
    #     # report = True,
    # )


    print('export duration_predictor.onnx ok!')
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')
    save_fp16_onnx(onnx_file)
    return pred_dur, input_tensor, expanded_indices, en, t_en, expanded_length

def export_text_encoder(model, input, output_dir):
    onnx_file = output_dir + "/" + "text_encoder.onnx"

    (en, style, expanded_indices, t_en, expanded_length) = input

    print(f"\n\n\n\nexport_text_encoder: {en.shape=} {style.shape=} {t_en.shape=} {expanded_indices.shape=}")

    class model_text_encoder(torch.nn.Module):
        def __init__(self, model):
            super(model_text_encoder, self).__init__()
            self.text_encoder = model.text_encoder
            self.predictor = model.predictor
            self.decoder = model.decoder

        def forward(self, en, style, expanded_indices, t_en, expanded_length):
            expanded_length_i64 = expanded_length.to(torch.int64).reshape(())
            actual_en = en[:, :expanded_length_i64, :]
            actual_indices = expanded_indices[:expanded_length_i64]
            F0_pred_actual, N_pred_actual = self.predictor.F0Ntrain(actual_en, style[:, 128:256])

            # The original line was:
            # asr = torch.repeat_interleave(t_en, pred_dur, dim=2)
            asr_actual = torch.index_select(t_en, 2, actual_indices)
            audio_actual = self.decoder(asr_actual, F0_pred_actual, N_pred_actual, style[:, 0:128]).reshape(-1)

            asr = torch.nn.functional.pad(asr_actual, (0, MAX_EXPANDED_LENGTH - asr_actual.shape[-1]))
            F0_pred = torch.nn.functional.pad(F0_pred_actual, (0, (MAX_EXPANDED_LENGTH * 2) - F0_pred_actual.shape[-1]))
            N_pred = torch.nn.functional.pad(N_pred_actual, (0, (MAX_EXPANDED_LENGTH * 2) - N_pred_actual.shape[-1]))
            audio = torch.nn.functional.pad(audio_actual, (0, MAX_AUDIO_LENGTH - audio_actual.shape[-1]))
            return audio, asr, F0_pred, N_pred

    with torch.no_grad():
        model_text_encoder_instance = model_text_encoder(model).eval()

        audio, asr, F0_pred, N_pred = model_text_encoder_instance(en, style, expanded_indices, t_en, expanded_length)
        print(f'\nmodel_text_encoder_instance: {en.shape=} {style.shape=} {expanded_indices.shape=} {t_en.shape=} {expanded_length=} {audio.shape=} {asr.shape=} {F0_pred.shape=} {N_pred.shape=}\n\n')
        torch.onnx.export(
            model_text_encoder_instance, 
            args =  (en, style, expanded_indices, t_en, expanded_length), 
            f = onnx_file, 
            export_params = True, 
            verbose = False, 
            input_names = [ 'en', 'style', 'expanded_indices', 't_en', 'expanded_length' ], 
            output_names = [ 'audio', 'asr', 'F0_pred', 'N_pred' ],
            opset_version = OPSET_VERSION, 
            do_constant_folding = True, 
            dynamo = False,
            external_data=False,
            # report = True,
        )

    print('export text_encoder.onnx ok!')
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')
    save_fp16_onnx(onnx_file)
    return audio, asr, F0_pred, N_pred

def export_onnx(model, output_dir):
    onnx_file = output_dir + "/" + "kokoro.onnx"

    # input_ids = torch.randint(1, 100, (502,)).numpy()
    # input_ids = torch.LongTensor([[0, *input_ids, 0]])
    # style = torch.randn(1, 256)
    # speed = torch.randint(1, 2, (1,)).int()
    # print(f'\n\nexport_onnx SIM: {input_ids.shape=} {style.shape=} {speed.shape=}'          )

    input_ids, style, speed = load_sample(model)
    input_length = torch.tensor(input_ids.shape[1], dtype=torch.int32)
    text_mask = torch.zeros(1, MAX_INPUT_LENGTH, dtype=torch.float32).to(input_ids.device)
    text_mask[0, :input_ids.shape[1]] = 1

    print(f'\n\nexport_onnx    : {input_ids.shape=} {style.shape=} {speed.shape=}')

    input_ids = torch.nn.functional.pad(input_ids, (0, MAX_INPUT_LENGTH - input_ids.shape[1]))

    d_en = export_bert(model.kmodel, (input_ids, text_mask), output_dir=output_dir)
    pred_dur, input_tensor, expanded_indices, en, t_en, expanded_length = export_duration_predictor(model.kmodel, (input_ids, d_en, style, input_length, speed), output_dir=output_dir)
    audio, asr, F0_pred, N_pred = export_text_encoder(model.kmodel, (en, style, expanded_indices, t_en, expanded_length), output_dir=output_dir)

    # Save audio to onnx_test.wav
    import scipy.io.wavfile as wavfile
    audio = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    expanded_length_value = int(expanded_length.item()) if isinstance(expanded_length, torch.Tensor) else int(expanded_length)
    audio = audio[: expanded_length_value * AUDIO_SAMPLES_PER_FRAME]
    wavfile.write(os.path.join(output_dir, 'onnx_test.wav'), 24000, (audio * 32767).astype('int16'))
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.savefig(os.path.join(output_dir, 'onnx_test.png'))
    plt.close()
    compare_native_vs_export_compatible(model, output_dir)
    inference_onnx_parts(model, output_dir)
    print('export kokoro.onnx ok!')

    # onnx_model = onnx.load(onnx_file)
    # onnx.checker.check_model(onnx_model)
    # print('onnx check ok!')



#########################################################################













def load_input_ids(pipeline, text):
    if pipeline.lang_code in 'ab':
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps, _ = pipeline.g2p(text)

    if len(ps) > 510:
        ps = ps[:510]

    input_ids = list(filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), ps)))
    print(f"text: {text} -> phonemes: {ps} -> input_ids: {input_ids}")
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids

def load_voice(pipeline, voice, phonemes):
    pack = pipeline.load_voice(voice).to('cpu')
    return pack[len(phonemes) - 1]

def load_sample(model):
    pipeline = KPipeline(lang_code='a', model=model.kmodel, device='cpu')
    text = DEFAULT_SAMPLE_TEXT
    voice = 'checkpoints/voices/af_heart.pt'

    # pipeline = KPipeline(lang_code='z', model=model.kmodel, device='cpu')
    # text = '''
    # 2月15日晚，猫眼专业版数据显示，截至发稿，《哪吒之魔童闹海》（或称《哪吒2》）今日票房已达7.8亿元，累计票房（含预售）超过114亿元。
    # '''
    # voice = 'checkpoints/voices/zf_xiaoxiao.pt'

    phonemes, input_ids = load_input_ids(pipeline, text)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])

    return input_ids, style, speed

def prepare_part_inputs(model, text=None):
    pipeline = KPipeline(lang_code='a', model=model.kmodel, device='cpu')
    voice = 'checkpoints/voices/af_heart.pt'
    phonemes, input_ids = load_input_ids(pipeline, text or DEFAULT_SAMPLE_TEXT)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])
    input_length = torch.tensor(input_ids.shape[1], dtype=torch.int32)
    text_mask = torch.zeros(1, MAX_INPUT_LENGTH, dtype=torch.float32)
    text_mask[0, :input_ids.shape[1]] = 1
    input_ids = torch.nn.functional.pad(input_ids, (0, MAX_INPUT_LENGTH - input_ids.shape[1]))
    return input_ids, style, speed, input_length, text_mask


def save_waveform_plot(audio, path, title):
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_audio_file(audio, path):
    import scipy.io.wavfile as wavfile

    wavfile.write(path, 24000, (audio * 32767).astype('int16'))


def compute_part_features(model, text=None):
    input_ids, style, speed, input_length, text_mask = prepare_part_inputs(model, text=text)
    input_ids_actual = input_ids[:, :input_length]
    text_mask = torch.ones((1, input_ids_actual.shape[1]), dtype=torch.float32, device=input_ids.device)

    with torch.no_grad():
        bert_dur = model.kmodel.bert(input_ids_actual)
        d_en = model.kmodel.bert_encoder(bert_dur).transpose(-1, -2)

        d = model.kmodel.predictor.text_encoder(d_en, style[:, 128:], text_mask)
        x, _ = model.kmodel.predictor.lstm(d)
        duration = model.kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur_actual = torch.round(duration).clamp(min=1).reshape(-1)
        pred_dur = torch.nn.functional.pad(pred_dur_actual, (0, MAX_INPUT_LENGTH - pred_dur_actual.shape[0]))

        input_tensor = d.transpose(-1, -2)
        valid_token_count = input_length.to(torch.int32).reshape(())
        boundaries = torch.cumsum(pred_dur_actual, dim=0)
        expanded_length = torch.clamp(boundaries[-1].to(torch.int32), max=MAX_EXPANDED_LENGTH)
        values = torch.arange(MAX_EXPANDED_LENGTH, device=pred_dur.device, dtype=torch.float32)
        expanded_indices = torch.sum(
            (boundaries.unsqueeze(1) <= values.unsqueeze(0)).to(torch.float32),
            dim=0,
        ).to(torch.int32)
        expanded_indices = torch.clamp(expanded_indices, max=valid_token_count - 1)
        actual_indices = expanded_indices[:expanded_length]
        actual_en = torch.index_select(input_tensor, 2, actual_indices)
        actual_en, _ = model.kmodel.predictor.shared(actual_en.transpose(-1, -2))
        en = torch.nn.functional.pad(actual_en, (0, 0, 0, MAX_EXPANDED_LENGTH - actual_en.shape[1]))
        t_en = model.kmodel.text_encoder(input_ids_actual, text_mask)
        t_en = torch.nn.functional.pad(t_en, (0, MAX_INPUT_LENGTH - t_en.shape[-1]))

    return input_ids, style, speed, input_length, pred_dur, expanded_indices, en, t_en, expanded_length


def run_text_encoder_variant(model, en, style, expanded_indices, t_en, expanded_length, export_compatible=False):
    with torch.no_grad():
        with torch.random.fork_rng():
            torch.manual_seed(0)
            with export_compatible_ops(export_compatible):
                actual_en = en[:, :expanded_length, :]
                actual_indices = expanded_indices[:expanded_length]
                F0_pred_actual, N_pred_actual = model.kmodel.predictor.F0Ntrain(actual_en, style[:, 128:256])
                asr_actual = torch.index_select(t_en, 2, actual_indices)
                audio_actual = model.kmodel.decoder(asr_actual, F0_pred_actual, N_pred_actual, style[:, 0:128]).reshape(-1)

                asr = torch.nn.functional.pad(asr_actual, (0, MAX_EXPANDED_LENGTH - asr_actual.shape[-1]))
                F0_pred = torch.nn.functional.pad(F0_pred_actual, (0, (MAX_EXPANDED_LENGTH * 2) - F0_pred_actual.shape[-1]))
                N_pred = torch.nn.functional.pad(N_pred_actual, (0, (MAX_EXPANDED_LENGTH * 2) - N_pred_actual.shape[-1]))
                audio = torch.nn.functional.pad(audio_actual, (0, MAX_AUDIO_LENGTH - audio_actual.shape[-1]))
    return audio, asr, F0_pred, N_pred


def render_clean_native_reference(output_dir, text=None):
    temp_root = tempfile.mkdtemp(prefix="kokoro-baseline-")
    repo_root = os.getcwd()
    temp_pkg_root = os.path.join(temp_root, "pkg")
    temp_kokoro_dir = os.path.join(temp_pkg_root, "kokoro")
    os.makedirs(temp_pkg_root, exist_ok=True)
    shutil.copytree(os.path.join(repo_root, "kokoro"), temp_kokoro_dir, dirs_exist_ok=True)

    for rel_path in ("kokoro/istftnet.py", "kokoro/custom_stft.py"):
        result = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        target = os.path.join(temp_pkg_root, rel_path)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)

    output_npy = os.path.join(output_dir, "native_reference_audio.npy")
    sample_text = text or DEFAULT_SAMPLE_TEXT
    script = f"""
import numpy as np
import torch
from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX

DEFAULT_SAMPLE_TEXT = {sample_text!r}

def load_input_ids(pipeline, text):
    if pipeline.lang_code in 'ab':
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps, _ = pipeline.g2p(text)
    if len(ps) > 510:
        ps = ps[:510]
    input_ids = list(filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), ps)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids

def load_voice(pipeline, voice, phonemes):
    pack = pipeline.load_voice(voice).to('cpu')
    return pack[len(phonemes) - 1]

kmodel = KModel(config='checkpoints/config.json', model='checkpoints/kokoro-v1_0.pth', disable_complex=True)
model = KModelForONNX(kmodel).eval()
pipeline = KPipeline(lang_code='a', model=model.kmodel, device='cpu')
phonemes, input_ids = load_input_ids(pipeline, DEFAULT_SAMPLE_TEXT)
style = load_voice(pipeline, 'checkpoints/voices/af_heart.pt', phonemes)
speed = torch.IntTensor([1])
torch.manual_seed(0)
with torch.no_grad():
    audio, _ = model(input_ids, style, speed)
np.save({output_npy!r}, audio.detach().cpu().numpy())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = temp_pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
    )
    return np.load(output_npy)


def compare_native_vs_export_compatible(model, output_dir, text=None):
    _, style, _, _, pred_dur, expanded_indices, en, t_en, expanded_length = compute_part_features(model, text=text)
    export_audio, _, export_f0, export_n = run_text_encoder_variant(
        model, en, style, expanded_indices, t_en, expanded_length, export_compatible=True
    )
    native_audio = render_clean_native_reference(output_dir, text=text).reshape(-1)

    expanded_length_value = int(expanded_length.item()) if isinstance(expanded_length, torch.Tensor) else int(expanded_length)
    audio_len = min(expanded_length_value * AUDIO_SAMPLES_PER_FRAME, native_audio.shape[0])

    native_audio = native_audio[:audio_len]
    export_audio = export_audio[:audio_len].detach().cpu().numpy()
    audio_diff = native_audio - export_audio

    export_f0 = export_f0.detach().cpu().numpy().reshape(-1)
    export_n = export_n.detach().cpu().numpy().reshape(-1)

    save_waveform_plot(native_audio, os.path.join(output_dir, 'native_reference_test.png'), 'Clean Native Reference Audio')
    save_waveform_plot(export_audio, os.path.join(output_dir, 'export_compatible_parts_test.png'), 'Export-Compatible Decoder Audio')
    save_audio_file(native_audio, os.path.join(output_dir, 'native_reference_test.wav'))
    save_audio_file(export_audio, os.path.join(output_dir, 'export_compatible_parts_test.wav'))

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(native_audio, label='native reference', alpha=0.9)
    axes[0].plot(export_audio, label='export-compatible', alpha=0.7)
    axes[0].set_title('Reference vs Export-Compatible Audio')
    axes[0].legend()
    axes[1].plot(audio_diff, color='tab:red')
    axes[1].set_title('Reference Minus Export-Compatible')
    axes[2].plot(export_f0[: min(4096, export_f0.shape[0])], label='export F0')
    axes[2].plot(export_n[: min(4096, export_n.shape[0])], label='export N', alpha=0.8)
    axes[2].set_title('Export-Compatible Predictions')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'native_vs_export_compatible_comparison.png'))
    plt.close(fig)

    audio_mae = float(np.mean(np.abs(audio_diff)))
    audio_max = float(np.max(np.abs(audio_diff)))
    audio_rmse = float(np.sqrt(np.mean(np.square(audio_diff))))

    print(
        "compare_native_vs_export_compatible:",
        "reference=native_reference_test.wav",
        f"audio_mae={audio_mae:.6f}",
        f"audio_rmse={audio_rmse:.6f}",
        f"audio_max={audio_max:.6f}",
        f"export_f0_mean={float(np.mean(np.abs(export_f0))):.6f}",
        f"export_n_mean={float(np.mean(np.abs(export_n))):.6f}",
    )

    return {
        "audio_mae": audio_mae,
        "audio_rmse": audio_rmse,
        "audio_max": audio_max,
    }


def run_onnx_session(session, feeds):
    input_names = {input_meta.name for input_meta in session.get_inputs()}
    return session.run(None, {
        name: value for name, value in feeds.items() if name in input_names
    })

def inference_onnx(model, output):
    onnx_file = output + "/" + "kokoro.onnx"
    session = ort.InferenceSession(onnx_file)

    input_ids, style, speed = load_sample(model)

    outputs = session.run(None, {
        'input_ids': input_ids.numpy(), 
        'style': style.numpy(), 
        'speed': speed.numpy(), 
    })

    output = torch.from_numpy(outputs[0])
    print(f'output: {output.shape}')
    print(output)

    # audio = output.numpy()
    # sd.play(audio, 24000)
    # sd.wait()
    audio = output.numpy()
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.savefig('audio.png')
    plt.close()

    import scipy.io.wavfile as wavfile
    wavfile.write('audio.wav', 24000, (audio * 32767).astype('int16'))

def inference_onnx_parts(model, output, text=None):
    bert_session = ort.InferenceSession(os.path.join(output, "bert.onnx"))
    duration_session = ort.InferenceSession(os.path.join(output, "duration_predictor.onnx"))
    text_encoder_session = ort.InferenceSession(os.path.join(output, "text_encoder.onnx"))

    input_ids, style, speed, input_length, text_mask = prepare_part_inputs(model, text=text)

    d_en = run_onnx_session(bert_session, {
        "input_ids": input_ids.numpy(),
        "text_mask": text_mask.numpy(),
    })[0]

    pred_dur, d, expanded_indices, en, t_en, expanded_length = run_onnx_session(duration_session, {
        "input_ids": input_ids.numpy(),
        "d_en": d_en,
        "style": style.numpy(),
        "input_length": input_length.numpy(),
        "speed": speed.numpy(),
    })

    audio, asr, F0_pred, N_pred = run_onnx_session(text_encoder_session, {
        "en": en,
        "style": style.numpy(),
        "expanded_indices": expanded_indices,
        "t_en": t_en,
        "expanded_length": expanded_length,
    })

    expanded_length_value = int(np.asarray(expanded_length).reshape(()))
    audio = audio[: expanded_length_value * AUDIO_SAMPLES_PER_FRAME]

    print(f"inference_onnx_parts: {pred_dur.shape=} {d.shape=} {expanded_indices.shape=} {en.shape=} {t_en.shape=} {expanded_length_value=} {audio.shape=} {asr.shape=} {F0_pred.shape=} {N_pred.shape=}")

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.savefig(os.path.join(output, 'onnx_parts_test.png'))
    plt.close()

    import scipy.io.wavfile as wavfile
    wavfile.write(os.path.join(output, 'onnx_parts_test.wav'), 24000, (audio * 32767).astype('int16'))

def check_model(model):
    input_ids, style, speed = load_sample(model)
    output, duration = model(input_ids, style, speed)

    print(f'output: {output.shape}')
    print(f'duration: {duration.shape}')
    print(output)

    audio = output.numpy()
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.savefig('audio.png')
    plt.close()
    import scipy.io.wavfile as wavfile
    wavfile.write('audio.wav', 24000, (audio * 32767).astype('int16'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export kokoro Model to ONNX", add_help=True)
    parser.add_argument("--inference", "-t", help="test kokoro.onnx model", action="store_true")
    parser.add_argument("--check", "-m", help="check kokoro model", action="store_true")
    parser.add_argument(
        "--config_file", "-c", type=str, default="checkpoints/config.json", help="path to config file"
    )
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="checkpoints/kokoro-v1_0.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="onnx", help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    kmodel = KModel(config=config_file, model=checkpoint_path, disable_complex=True)
    model = KModelForONNX(kmodel).eval()

    if args.inference:
        inference_onnx(model, output_dir)
    elif args.check:
        check_model(model)
    else:
        export_onnx(model, output_dir)
