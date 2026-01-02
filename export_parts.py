import argparse
import os
from matplotlib import style
import torch
import onnx
import onnxruntime as ort
import sounddevice as sd
from torch.export import Dim

from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX
import matplotlib

##########################################################################
OPSET_VERSION = 19



def export_bert(model, input, output_dir):
    onnx_file = output_dir + "/" + "bert.onnx"

    (input_ids, style, speed) = input

    batch_size = Dim.STATIC #Dim("batch_size", min=1, max=32)
    input_len = Dim("seq_length", min=2, max=510)

    class model_bert(torch.nn.Module):
        def __init__(self, model):
            super(model_bert, self).__init__()
            self.bert = model.bert
            self.bert_encoder = model.bert_encoder

        def forward(self, input_ids):
            bert_dur = self.bert(input_ids)
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            return d_en

    model_bert_instance = model_bert(model).eval()

    d_en = model_bert_instance(input_ids)
    print(f'bert output d_en: {d_en.shape}')

    torch.onnx.export(
        model_bert_instance, 
        args =  (input_ids,), 
        f = onnx_file, 
        export_params = True, 
        verbose = False, 
        input_names = [ 'input_ids' ], 
        output_names = [ 'd_en' ],
        opset_version = OPSET_VERSION, 
        dynamic_shapes = { 'input_ids': {0: batch_size, 1: input_len} },
        do_constant_folding = True, 
        dynamo = True,
        external_data=False,
        # report = True,
    )

    print('export bert.onnx ok!')
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')
    return d_en

def export_duration_predictor(model, input, output_dir):
    onnx_file = output_dir + "/" + "duration_predictor.onnx"

    (input_ids, d_en, style, speed) = input

    batch_size = Dim.STATIC #Dim("batch_size", min=1, max=32)
    input_len = Dim("seq_length", min=2, max=510)
    feature_dim = Dim.STATIC #Dim("feature_dim", min=64, max=512)
    
    class model_duration_predictor(torch.nn.Module):
        def __init__(self, model):
            super(model_duration_predictor, self).__init__()
            self.predictor = model.predictor
            self.text_encoder = model.text_encoder

        def forward(self, input_ids, d_en, style, speed):
            d = self.predictor.text_encoder(d_en, style[:, 128:])
            x, _ = self.predictor.lstm(d)
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).squeeze()

            input_tensor = d.transpose(-1, -2)

            boundaries = torch.cumsum(pred_dur, dim=0)
            # print(f'\n\n\nboundaries: {boundaries=}\n\n\n')
            print(f'\n\n\nboundaries: {boundaries.shape=} {boundaries[-1]=}\n\n\n')
            values = torch.arange(boundaries[-1], device=pred_dur.device)
            expanded_indices = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
            en = torch.index_select(input_tensor, 2, expanded_indices)
            en, _ = self.predictor.shared(en.transpose(-1, -2)) #moved lstm here from F0Ntrain to make dyanmo export of F0Ntrain possible
            t_en = self.text_encoder(input_ids)

            return pred_dur, input_tensor, expanded_indices, en, t_en

    with torch.no_grad():
        model_duration_instance = model_duration_predictor(model).eval()

        pred_dur, input_tensor, expanded_indices, en, t_en = model_duration_instance(input_ids, d_en, style, speed)
        print(f'\n\n\nduration predictor output pred_dur: {pred_dur.shape=} {en.shape=} {t_en.shape=} {expanded_indices=}\n\n\n')

        torch.onnx.export(
            model_duration_instance, 
            args =  (input_ids, d_en, style, speed), 
            f = onnx_file, 
            export_params = True, 
            verbose = False, 
            input_names = [ 'input_ids', 'd_en', 'style', 'speed' ], 
            output_names = [ 'pred_dur', 'd', 'expanded_indices', 'en', 't_en' ],
            opset_version = OPSET_VERSION, 
            dynamic_axes = { 
                'input_ids': {1: 'input_len'},
                'd_en': {2: 'seq_length'}, 
                'pred_dur': {0: 'seq_length'},
                'd': {1: 'seq_length'},
            },
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
    return pred_dur, input_tensor, expanded_indices, en, t_en

def export_text_encoder(model, input, output_dir):
    onnx_file = output_dir + "/" + "text_encoder.onnx"

    (en, style, expanded_indices, t_en) = input

    print(f"\n\n\n\nexport_text_encoder: {en.shape=} {style.shape=} {t_en.shape=} {expanded_indices.shape=}")

    class model_text_encoder(torch.nn.Module):
        def __init__(self, model):
            super(model_text_encoder, self).__init__()
            self.text_encoder = model.text_encoder
            self.predictor = model.predictor
            self.decoder = model.decoder

        def forward(self, en, style, expanded_indices, t_en):
            F0_pred, N_pred = self.predictor.F0Ntrain(en, style[:, 128:256])

            # The original line was:
            # asr = torch.repeat_interleave(t_en, pred_dur, dim=2)
            asr = torch.index_select(t_en, 2, expanded_indices)
            # audio = None
            audio = self.decoder(asr, F0_pred, N_pred, style[:, 0:128]).squeeze()        
            return audio, asr, F0_pred, N_pred

    with torch.no_grad():
        model_text_encoder_instance = model_text_encoder(model).eval()

        audio, asr, F0_pred, N_pred = model_text_encoder_instance(en, style, expanded_indices, t_en)

        batch_size = Dim.STATIC 
        en_len = Dim("en", min=2) #, max=30000)
        input_len = Dim("seq_length", min=2, max=510)
        style_len = Dim.STATIC 
        feature_dim = Dim.STATIC #Dim("feature_dim", min=64, max=512)
        torch.onnx.export(
            model_text_encoder_instance, 
            args =  (en, style, expanded_indices, t_en,), 
            f = onnx_file, 
            export_params = True, 
            verbose = True, 
            input_names = [ 'en', 'style', 'expanded_indices', 't_en' ], 
            output_names = [ 'audio', 'asr', 'F0_pred', 'N_pred' ],
            opset_version = OPSET_VERSION, 
            dynamic_shapes = { 
                'en': {0: batch_size, 1: en_len, 2: feature_dim}, 
                'style': {0: batch_size, 1: style_len},
                'expanded_indices': {0: en_len},
                't_en': {0: batch_size, 1: feature_dim, 2: input_len},
            },
            do_constant_folding = True, 
            dynamo = True,
            external_data=False,
            report = True,
        )

    print('export text_encoder.onnx ok!')
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')
    return audio, asr, F0_pred, N_pred

def export_onnx(model, output_dir):
    onnx_file = output_dir + "/" + "kokoro.onnx"

    # input_ids = torch.randint(1, 100, (502,)).numpy()
    # input_ids = torch.LongTensor([[0, *input_ids, 0]])
    # style = torch.randn(1, 256)
    # speed = torch.randint(1, 2, (1,)).int()
    # print(f'\n\nexport_onnx SIM: {input_ids.shape=} {style.shape=} {speed.shape=}'          )

    input_ids, style, speed = load_sample(model)
    print(f'\n\nexport_onnx    : {input_ids.shape=} {style.shape=} {speed.shape=}')

    d_en = export_bert(model.kmodel, (input_ids, style, speed), output_dir=output_dir)
    pred_dur, input_tensor, expanded_indices, en, t_en = export_duration_predictor(model.kmodel, (input_ids, d_en, style, speed), output_dir=output_dir)
    audio, asr, F0_pred, N_pred = export_text_encoder(model.kmodel, (en, style, expanded_indices, t_en), output_dir=output_dir)

    # Save audio to onnx_test.wav
    import scipy.io.wavfile as wavfile
    audio = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    wavfile.write(os.path.join(output_dir, 'onnx_test.wav'), 24000, (audio * 32767).astype('int16'))
    print('export kokoro.onnx ok!')

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('onnx check ok!')



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
    text = '''
    In today's fast-paced tech world, building software applications has never been easier — thanks to AI-powered coding assistants.'
    '''
    text = '''
    The sky above the port was the color of television, tuned to a dead channel.
    '''
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
