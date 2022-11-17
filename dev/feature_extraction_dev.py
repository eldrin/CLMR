from typing import Any
import argparse
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np
import pandas as pd

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from clmr.datasets import get_dataset
from clmr.datasets.audio import AUDIO
from clmr.models import SampleCNNS, SampleCNN
from clmr.utils import load_encoder_checkpoint, yaml_config_hook

from tqdm import tqdm


MODEL_CLASSES = {
    'small': SampleCNNS,
    'basic': SampleCNN
}


def parse_arguments() -> tuple[argparse.Namespace, dict[str, Any]]:
    """
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)

    parser0 = argparse.ArgumentParser(parents=[parser])
    parser0.add_argument('config_fn', type=str,
                         help="the filename of the extraction configration file")
    parser0.add_argument('checkpoint_fn', type=str,
                         help="the filename of the model checkpoint file")
    parser0.add_argument('-m', '--model-type', type=str, default='small',
                         choices=MODEL_CLASSES.keys(),
                         help=("size of the model class that is used for "
                               "the feature extraction."))
    parser0.add_argument('-p', '--path', type=str, default='./',
                         help="the path where the result is stored.")
    args = parser0.parse_args()

    config = yaml_config_hook(args.config_fn)
    for k, v in config.items():
        parser0.add_argument(f"--{k}", default=v, type=type(v))
    args = parser0.parse_args()

    return args, config



def extract(
    config: dict[str, Any],
    path: str,  # out_path
    model_type: str,
    checkpoint_fn: str,
    seed: int,
    audio_length: int,
    dataset_dir: str,
    dataset: str,
    accelerator: str,
    n_classes: int = 10  # seems not used
):
    """ extract features from the pretrained CLMR models

    TODO: can we detect and raise error when specified model type doesn't match
          to the checkpoint?

    TODO: also we probably should refactor this loooong function

    Args:
        config: configuration dictionary used for the extraction procedure
        path: output path where the extracted feature stored
        model_type: type of the model class used for the extraction
        checkpoint_fn: filename of the saved pre-trained model parameters
        seed: random seed to populate the generator
        audio_length: length of the input audio signal
        dataset_dir: path where the dataset is stored
        dataset: the dataset class {'gtzan', 'echonest', 'mtat'}
        accelerator: the type of computing resource {'cpu', 'cuda', 'cuda:0', 'cuda:1', ... }
        n_clsseses: it is not used, but here for feeding default value of some modules

    """
    # set some stuffs
    dataset_name = config['dataset_name']
    out_fn = Path(path) / f'{dataset_name}_feature.npz'
    out_fn.parent.mkdir(exist_ok=True, parents=True)

    pl.seed_everything(seed)

    # -------------
    # dataloaders
    # -------------
    dataset_: AUDIO = get_dataset(dataset, dataset_dir, subset='train')

    # ---------
    # encoder
    # ---------
    encoder = MODEL_CLASSES[model_type](
        strides = [3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised = False,
        out_dim = n_classes
    )
    state_dict = load_encoder_checkpoint(checkpoint_fn,
                                         n_classes,
                                         encoder.fc.in_features)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.to(accelerator)
    # encoder.freeze()  # ?

    # -----------
    # extraction
    # -----------
    representations = []
    for j in tqdm(range(len(dataset_)), ncols=80):
        with torch.no_grad():
            audio, _ = dataset_[j]
            batch = torch.split(audio, audio_length, dim=1)
            if len(batch) < 2:
                # tmp = batch
                x = torch.zeros((1, audio_length)).to(accelerator)
                x[:, :batch[0].shape[1]] = batch[0]
                batch = (x,)
            else:
                batch = batch[:-1]
            batch = torch.cat(batch)
            batch = batch.unsqueeze(dim=1)

            if batch.shape[0] > 128:
                h0 = []
                for i in range(int(batch.shape[0] / 48) + 1):
                    batch_ = batch[i*48:(i+1)*48]
                    batch_ = batch_.to(accelerator)
                    h0_ = encoder.sequential(batch_)
                    h0.append(h0_)
                h0 = torch.cat(h0, dim=0)
            else:
                batch = batch.to(accelerator)
                h0 = encoder.sequential(batch)
        representations.append(h0.mean(0)[None])

    if len(representations) > 1:
        representations = torch.cat(representations, dim=0)
    else:
        representations = representations[0]
    representations = representations[:, :, 0].detach().cpu().numpy()
    print('feature shape:', representations.shape)

    # ----------
    # packaging
    # ----------
    if config['dataset_name'] == 'gtzan':
        # compute ids
        ids = []
        for fn in dataset_.fl:
            i = Path(fn).stem
            ids.append(f'{i}.au')
        ids = np.array(ids)

    elif config['dataset_name'] == 'mtat':
        # load the metadata and build the fn -> clip_id map
        metadata = pd.read_csv(config['dataset_metadata_path'], sep='\t')

        # seems type hints somehow is confused this as TextFileReader
        # without this explicit wrapping :(
        metadata = pd.DataFrame(metadata)

        fn2clipid = {
            Path(fn).stem + '.wav': clipid
            for fn, clipid
            in metadata[['mp3_path', 'clip_id']].dropna().values
        }
        ids = []
        mat_idx = []
        for i, fn in enumerate(dataset_.fl):
            fn_ = Path(fn).name
            if fn_ not in fn2clipid:
                continue
            clipid = fn2clipid[fn_]
            ids.append(f'{clipid}')
            mat_idx.append(i)
        ids = np.array(ids)
        representations = representations[mat_idx]

    elif config['dataset_name'] == 'echonest':
        # compute ids
        ids = []
        for fn in dataset_.fl:
            i = Path(fn).stem
            ids.append(i)
        ids = np.array(ids)

    else:
        raise ValueError('[ERROR] only `gtzan` and `mtat` is supported yet!')

    print('feature shape:', representations.shape)

    np.savez(
        out_fn,
        feature = representations,
        ids = ids,
        dataset = np.array(config['dataset_name']),
        model_class = np.array('CLMR'),
        model_filename = np.array(checkpoint_fn)
    )


def main():
    """
    """
    # parse argument
    args, config = parse_arguments()

    # procedure
    extract(
        config,
        path = args.path,
        model_type = args.model_type,
        checkpoint_fn = args.checkpoint_fn,
        seed = args.seed,
        audio_length = args.audio_length,
        dataset_dir = args.dataset_dir,
        dataset = args.dataset,
        accelerator = args.accelerator
    )


if __name__ == "__main__":
    main()
