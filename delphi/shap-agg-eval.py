import os
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
import shap
from tqdm.autonotebook import tqdm

from model import DelphiConfig, Delphi
from utils import get_batch, get_p2i
from utils import shap_custom_tokenizer, shap_model_creator


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate SHAP values for Delphi model')
    parser.add_argument('--checkpoint_dir', type=str, default='Delphi-2M',
                        help='Directory containing the model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/ukb_simulated_data',
                        help='Directory containing train.bin and val.bin')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of people to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu, cuda, mps, etc.)')
    parser.add_argument('--output', type=str, default='shap_agg.pickle',
                        help='Output pickle path')
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()


def load_model(checkpoint_dir, device):
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    conf = DelphiConfig(**checkpoint['model_args'])
    model = Delphi(conf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(device)
    return model, conf


def get_person(idx, val, val_p2i, id_to_token, block_size, device):
    x, a, _, b = get_batch(
        [idx], val, val_p2i,
        select='left', block_size=block_size,
        device=device, padding='random',
        cut_batch=True,
    )
    valid = a[0] > -1
    x_valid, a_valid = x[0][valid], a[0][valid]

    person = [(id_to_token[tok.item()], age.item()) for tok, age in zip(x_valid, a_valid)]
    return person, a_valid, b[0, -1]


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed(args.seed)

    model, conf = load_model(args.checkpoint_dir, args.device)

    delphi_labels = pd.read_csv('delphi_labels_chapters_colours_icd.csv')

    val = np.fromfile(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint32).reshape(-1, 3)
    val_p2i = get_p2i(val)

    id_to_token = delphi_labels['name'].to_dict()
    token_to_id = {v: k for k, v in id_to_token.items()}

    n_people = args.n if args.n is not None else len(val_p2i)

    shaply_val = []

    for person_idx in tqdm(range(n_people)):
        try:
            person, ages_tensor, target_time = get_person(
                person_idx, val, val_p2i, id_to_token, conf.block_size, args.device)
            time_passed = (target_time - ages_tensor).cpu().detach().numpy()

            person_tokens = [p[0] for p in person]
            person_ages = [p[1] for p in person]
            person_tokens_ids = [token_to_id[t] for t in person_tokens]

            masker = shap.maskers.Text(
                shap_custom_tokenizer, output_type='str',
                mask_token='10000', collapse_mask_token=False,
            )
            model_shap = shap_model_creator(
                model, delphi_labels.index.values,
                person_tokens_ids, person_ages, args.device,
            )
            explainer = shap.Explainer(model_shap, masker, output_names=delphi_labels['name'].values)

            shap_input = ' '.join(str(token_to_id[t]) for t in person_tokens)
            shap_values = explainer([shap_input])
            shap_values.data = np.array([
                [f"{name}({age / 365:.1f}) " for name, age in person]
            ])

            shaply_val.append((
                person_tokens_ids,
                shap_values.values.astype(np.float16),
                time_passed,
                [person_idx] * len(person_tokens_ids),
            ))
        except Exception as e:
            print(repr(e))

    all_tokens = np.concatenate([i[0] for i in shaply_val])
    all_values = np.concatenate([i[1] for i in shaply_val], axis=1)[0]
    all_times_passed = np.concatenate([i[2] for i in shaply_val], axis=0)
    all_people = np.concatenate([i[3] for i in shaply_val])

    with open(args.output, 'wb') as f:
        pickle.dump({
            'tokens': all_tokens,
            'values': all_values,
            'times': all_times_passed,
            'model': args.checkpoint_dir,
            'people': all_people,
        }, f)

    print(f'Saved SHAP aggregation to {args.output}')


if __name__ == '__main__':
    main()
