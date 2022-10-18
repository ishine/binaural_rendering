import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from moyu.utils.yaml import read_yaml


def build_render(args):
    if args.use_hrir:
        if args.config_yaml is not None:
            print("Disable NN rendering since SR-HRIR is applied.")
        from trivial.infer import decode_ambisonic_to_binaural
        return decode_ambisonic_to_binaural
    elif args.config_yaml is not None and args.checkpoint_path is not None:
        config = read_yaml(args.config_yaml)
        checkpoint_path = args.checkpoint_path

        model_type = config["train"]["model_type"]
        from moyu.model.ambisonic_to_binaural import get_model_class

        Model = get_model_class(model_type)

        model_kwds = {}
        if "model" in config and "kwds" in config["model"]:
            model_kwds = config["model"]["kwds"]

        model = Model(**model_kwds)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
       
        model.load_state_dict(checkpoint["model"])
        device="cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        from moyu.model.ambisonic_to_binaural.separator import Separator

        return Separator(
            model = model,
            segment_samples=3*48000,
            batch_size=4,
            device=device
        )
    else:
        raise Exception("Unexpected arguments")


def infer_file(args):
    separator = build_render(args)

    input, sr = sf.read(args.input_path)
    # (samples, channels)
    if args.use_hrir:
        input = torch.tensor(input).T[None, ...]
        output = separator(input)[0].T
        output = np.array(output)
        # output, sr = sf.read(args.input_path)
    else:
        output = separator.separate({"waveform": input.T}).T

    if args.target_path is not None:
        # Calculate metrics
        # (samples, channels)
        target, _ = sf.read(args.target_path)
    
        from moyu.utils.calculate import calculate_lsd, calculate_sdr

        sdr = calculate_sdr(output, target)
        lsd = calculate_lsd(output.T, target.T)

        print("SDR: {:.4f}, lsd: {:.4f}".format(sdr, lsd))

    # Write output
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(file=output_path.resolve(), data=output, samplerate=sr)
        print("Write infer result to: {}".format(output_path))

def infer_dir(args):
    separator = build_render(args)
    
    input_dir = Path(args.input_dir)

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
    
    target_dir = Path(args.target_dir) if args.target_dir is not None else None
    
    from moyu.utils.calculate import calculate_lsd, calculate_sdr

    sdrs = []
    lsds = []
    for input_path in input_dir.iterdir():

        input, sr = sf.read(input_path)
        # (samples, channels)
        if args.use_hrir:
            output = separator(input)
        else:
            output = separator.separate({"waveform": input.T}).T

        if args.target_dir is not None:
            # Calculate metrics
            # (samples, channels)
            target_path = target_dir / input_path.name.replace("AmbiX", "Binaural")
            target, _ = sf.read(target_path)
                
            sdr = calculate_sdr(output, target)
            lsd = calculate_lsd(output.T, target.T)

            sdrs.append(sdr)
            lsds.append(lsd)

        # Write output
        if output_dir is not None:
            output_path = output_dir / input_path.name
            sf.write(file=output_path.resolve(), data=output, samplerate=sr)
    
    print('Avg SDR: {:.3f} LSD: {:.3f}'.format(np.mean(sdrs), np.mean(lsds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    
    parser_file = subparsers.add_parser("infer_file")
    parser_file.add_argument("--input_path", type=str)
    parser_file.add_argument("--output_path", type=str)
    parser_file.add_argument("--target_path", nargs="?", default=None, type=str)
    parser_file.add_argument("--use_hrir", action="store_true", default=False)
    parser_file.add_argument("--config_yaml", nargs="?", default=None, type=str)
    parser_file.add_argument("--checkpoint_path", nargs="?", default=None, type=str)


    parser_dir = subparsers.add_parser("infer_dir")
    parser_dir.add_argument("--input_dir", type=str)
    parser_dir.add_argument("--output_dir", type=str)
    parser_dir.add_argument("--target_dir", type=str)
    parser_dir.add_argument("--use_hrir", action="store_true", default=False)
    parser_dir.add_argument("--config_yaml", nargs="?", default=None, type=str)
    parser_dir.add_argument("--checkpoint_path", nargs="?", default=None, type=str)

    args = parser.parse_args()
    
    if args.mode == "infer_file":
        infer_file(args)
    elif args.mode == "infer_dir":
        infer_dir(args)
    else:
        raise NotImplementedError
    