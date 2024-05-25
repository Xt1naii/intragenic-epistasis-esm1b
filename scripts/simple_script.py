import argparse
import csv
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import errno
import os

import pandas as pd
import numpy as np
import torch
from Bio import SeqIO

from get_fasta import get_fasta
from esm_variants_utils import get_wt_LLR, load_esm_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setStream(sys.stderr)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

MODEL = 'esm1b_t33_650M_UR50S'

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')


def get_parser():
    parser = argparse.ArgumentParser(description="a simple script for getting LLR", add_help=False)
    parser.add_argument("-f", "--fasta_file", dest="fasta_file", required=False,
                    help="Input fasta file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-i", "--id_file", dest="id_file", required=False,
                    help="Input id file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))

    parser.add_argument(
        '-v', '--version', action='version',
                    version='1.1', help="Show program's version number and exit."
    )
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Show this help message and exit.'
    )
    parser.add_argument(
        '-o', '--output', required=False,
        help='Directs the output to a name of your choice',
        default='result')
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    logger.info('Parsing fasta file')
    fasta_sequences = SeqIO.parse(args.fasta_file,'fasta')
    data = {}
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        print(name)
        data[name] = sequence

    logger.info('Parsing id file')
    with args.id_file as file:
        id_data = file.read().splitlines() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Downloading model.. wait a sec')
    model, alphabet, batch_converter, _ = load_esm_model(model_name=MODEL, device='cuda')

    with open(f"{args.output}.native.llr.tsv", 'w') as f, open(f"{args.output}.mean.var.tsv", 'w') as g:
        for id in tqdm(data):
            logger.info('Downloading information from uniprot')
            result = get_fasta(id)
            input_df = pd.DataFrame(result, index=[0])
            _, LLR = get_wt_LLR(input_df, model, alphabet, batch_converter, device, silent=False)

            df = LLR[0].T
            df = df.stack().reset_index()
            df.columns = ['row','column','esm1b_score']
            df = df.round(3)
            df['uniprot_id'] = id
            df = df.reset_index(drop=True)
            df=df.reindex(columns=['uniprot_id', 'row', 'column', 'esm1b_score'])
            df.to_csv(f, index=False, header=True)

            llr = LLR[0].copy()
            llr = llr.replace(0.0, np.NaN)
            mv_res = {}
            mv_res['uniprot_id'] = id
            mv_res['mean_global'] = np.nanmean(llr)
            mv_res['var_global'] = np.nanvar(llr)
            mv_res = pd.DataFrame(mv_res, index=[0])
            llr_aa = pd.DataFrame({'aa':llr.columns})
            mv_aa_mean = llr.mean(0).values
            mv_aa_var = llr.var(0).values
            mv_aa_mean = pd.DataFrame({'aa_mean':mv_aa_mean})
            mv_aa_var = pd.DataFrame({'aa_var':mv_aa_var})
            mv_res = pd.concat([mv_res, llr_aa, mv_aa_mean, mv_aa_var], ignore_index=True, axis=1)
            mv_res.columns = ['uniprot_id','mean_global','var_global', 'aa', 'mean_local', 'var_local']
            mv_res.to_csv(g, index=False, header=True)


if __name__ == '__main__':
    main()
