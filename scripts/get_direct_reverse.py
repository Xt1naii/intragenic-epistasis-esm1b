import argparse
import requests
import numpy as np
import pandas as pd
import torch
import csv
from pathlib import Path
from csv import writer

from get_fasta import get_fasta
from esm_variants_utils import get_wt_LLR, load_esm_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Download the model.. wait a sec")
model,alphabet,batch_converter,repr_layer = load_esm_model(model_name='esm1b_t33_650M_UR50S',device='cuda')

def validate_file(arg):
    if (file := Path(arg)).is_file():
        return file
    else:
        raise FileNotFoundError(arg)

def get_parser():
    parser = argparse.ArgumentParser(description="a script for getting LLR")
    parser.add_argument(
      "--file", type=validate_file, help="Input file path", required=True
    )
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    print('Parsing the file...')
    with open(args.file, 'r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]

    print(len(data))
    for prot in data:
        print("Downloading information from uniprot")
        result = get_fasta(prot.get('uniprot_id'))

        if prot.get('aa_change')[0] == result['seq'][int(prot.get('aa_change')[1:-1]) - 1]:
            print(f"Amino acid {prot.get('aa_change')[0]} at position {prot.get('aa_change')[1:-1]} exists! OK")
            print("Trying to get LLR scores for protein")
            input_df = pd.DataFrame(result, index=[0])
            _, LLR = get_wt_LLR(input_df, model, alphabet, batch_converter, device, silent=False)
            res_list = [prot.get('uniprot_id'), prot.get('aa_change')]
            res_list.append(round(LLR[0].iloc[:, int(prot.get('aa_change')[1:-1]) - 1].loc[prot.get('aa_change')[-1]], 3))


            new_change = prot.get('aa_change')[-1:] + prot.get('aa_change')[1:-1] + prot.get('aa_change')[:1]
            print(new_change)
            res_list.append(new_change)
            seq_modified = (result['seq'][:int(new_change[1:-1]) - 1] 
                            + new_change[0] + result['seq'][int(new_change[1:-1]) - 1 + 1:])
            result['seq'] = seq_modified
            input_df = pd.DataFrame(result, index=[0])
            _, LLR2 = get_wt_LLR(input_df, model, alphabet, batch_converter, device, silent=False)

            res_list.append(round(LLR2[0].iloc[:, int(new_change[1:-1]) - 1].loc[new_change[-1]], 3))
            res_list.append(prot.get('clinvar_label'))
            with open("check_dir_indir.all.llr.tsv", 'a') as f:
                writer_object = writer(f)
                writer_object.writerow(res_list)
                f.close()
        else:
            print(f"There is some problem with your fasta or mutation. Amino acid doesnt exist! Not OK")

if __name__ == '__main__':
    main()