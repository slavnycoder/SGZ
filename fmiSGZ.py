import argparse
import csv
import logging
import os
import sys

import numpy as np
from scipy.stats import binom
from scipy.stats import binom_test

__author__ = 'James Sun'
__contact__ = 'jsun@foundationmedicine.com'
__version__ = '1.0.0'
__doc__ = '''FMI SGZ method to evaluate CNA origin and zygosity.'''

ALPHA = 0.01


# Functions for the CNA model
def copy_number_to_log_ratio_base_level(purity, base_level, cn):
    return np.log2((purity * cn + 2 * (1 - purity)) / base_level)


def sgz_core(cna_data, short_variants):
    results = []
    for short_variant in short_variants:
        if short_variant.chr_.upper() in ['X', 'Y']:
            continue
        short_variant.frequency = min(short_variant.frequency, 1.0)

        seg_count = 0
        clonality = 'NA'
        in_tumor = 'NA'
        # sg for somatic/germline
        sg_status = 'nocall_segmentMissing'

        af_g1 = af_g2 = af_s1 = af_s2 = np.nan
        purity = segment_copy_number = minor_allele_copy_number = np.nan
        error_lr = error_maf = np.nan
        max_prob_germline = max_prob_somatic = logodds = np.nan
        for segment in cna_data:
            if segment.chr_ == short_variant.chr_:
                if segment.start <= short_variant.position <= segment.end:
                    seg_count += 1

                    purity = segment.purity
                    segment_copy_number = segment.copy_number
                    minor_allele_copy_number = segment.num_ma_tumor_pred
                    base_level = segment.base_level

                    # maf for minor allele freq
                    model_maf = segment.maf_pred
                    data_maf = segment.seg_maf
                    error_maf = abs(model_maf - data_maf)

                    # lr for log-ratio
                    model_lr = copy_number_to_log_ratio_base_level(purity, base_level, segment_copy_number)
                    data_lr = segment.seg_lr
                    error_lr = abs(model_lr - data_lr)
                    if minor_allele_copy_number == 0:
                        minor_allele_copy_number = segment_copy_number - minor_allele_copy_number
                    af_g1 = (purity * minor_allele_copy_number + 1 * (1 - purity)) / (purity * segment_copy_number + 2 * (1 - purity))
                    af_s1 = (purity * minor_allele_copy_number + 0 * (1 - purity)) / (purity * segment_copy_number + 2 * (1 - purity))
                    p_g1 = binom_test(round(short_variant.depth * short_variant.frequency), short_variant.depth, af_g1)
                    p_s1 = binom_test(round(short_variant.depth * short_variant.frequency), short_variant.depth, af_s1)
                    if segment_copy_number - minor_allele_copy_number != minor_allele_copy_number:
                        af_g2 = (purity * (segment_copy_number - minor_allele_copy_number) + 1 * (1 - purity)) / \
                                (purity * segment_copy_number + 2 * (1 - purity))
                        p_g2 = binom_test(round(short_variant.depth * short_variant.frequency), short_variant.depth, af_g2)
                        if segment_copy_number - minor_allele_copy_number != 0:
                            af_s2 = (purity * (segment_copy_number - minor_allele_copy_number) + 0 * (1 - purity)) / \
                                    (purity * segment_copy_number + 2 * (1 - purity))
                            p_s2 = binom_test(round(short_variant.depth * short_variant.frequency), short_variant.depth, af_s2)
                        else:
                            af_s2 = p_s2 = np.nan
                    else:
                        af_g2 = af_s2 = p_g2 = p_s2 = np.nan

                    max_prob_germline = max(p_g1, p_g2)
                    max_prob_somatic = max(p_s1, p_s2)

                    if max_prob_somatic == 0:
                        logodds = np.inf
                    elif max_prob_germline == 0:
                        logodds = -np.inf
                    else:
                        logodds = np.log10(max_prob_germline) - np.log10(max_prob_somatic)
                    if 'CYP2D6' in short_variant.mutation:
                        sg_status = 'nocall_CYP2D6'
                    elif 'HLA' in short_variant.mutation:
                        sg_status = 'nocall_HLA'
                    elif short_variant.frequency < 0.05 and 0.2 < purity < 0.9:
                        sg_status = 'subclonal somatic'
                        clonality = 'subclonal'
                        in_tumor = 'in tumor'
                    elif np.isnan(segment_copy_number) or np.isnan(minor_allele_copy_number):
                        sg_status = 'ambiguous_CNA_model'
                    elif purity > 0.95:
                        sg_status = 'nocall_purity > 95%'
                    elif short_variant.frequency > 0.95:
                        sg_status = 'germline'
                        clonality = 'clonal'
                        in_tumor = 'in tumor'
                    elif error_maf > 0.06 or error_lr > 0.4:
                        sg_status = 'ambiguous_CNA_model'
                    elif max_prob_germline > ALPHA > max_prob_somatic:
                        if logodds < 2:
                            sg_status = 'probable germline'
                        else:
                            sg_status = 'germline'
                        clonality = 'clonal'
                        if np.nanargmax([p_g1, p_g2]) == 0:
                            in_tumor = 'in tumor'
                        else:
                            minor_allele_copy_number = segment_copy_number - minor_allele_copy_number
                            if minor_allele_copy_number == 0:
                                in_tumor = 'not in tumor'
                    elif max_prob_somatic > ALPHA > max_prob_germline:
                        if logodds > -2:
                            sg_status = 'probable somatic'
                        else:
                            sg_status = 'somatic'
                        clonality = 'clonal'
                        in_tumor = 'in tumor'

                        if np.nanargmax([p_s1, p_s2]) == 1:
                            minor_allele_copy_number = segment_copy_number - minor_allele_copy_number
                    elif max_prob_germline > ALPHA and max_prob_somatic > ALPHA:
                        sg_status = 'ambiguous_both_G_and_S'
                    elif max_prob_germline < ALPHA and max_prob_somatic < ALPHA:
                        min_soma_eaf = min(af_s1, af_s2)
                        min_germ_eaf = min(af_g1, af_g2)
                        if purity >= 0.3 \
                                and short_variant.frequency < 0.25 \
                                and short_variant.frequency < min_soma_eaf / 1.5 \
                                and min_soma_eaf <= min_germ_eaf:
                            sg_status = 'subclonal somatic'
                            clonality = 'subclonal'
                            in_tumor = 'in tumor'
                        elif purity >= 0.3 \
                                and short_variant.frequency < 0.25 \
                                and short_variant.frequency < min_germ_eaf / 2.0 \
                                and min_germ_eaf < min_soma_eaf:
                            sg_status = 'subclonal somatic'
                            clonality = 'subclonal'
                            in_tumor = 'in tumor'
                        elif logodds < -5 and max_prob_somatic > 1e-10:
                            sg_status = 'somatic'
                            clonality = 'clonal'
                            in_tumor = 'in tumor'
                            if np.nanargmax([p_s1, p_s2]) == 1:
                                minor_allele_copy_number = segment_copy_number - minor_allele_copy_number
                        elif logodds > 5 and max_prob_germline > 1e-4:
                            sg_status = 'germline'
                            clonality = 'clonal'
                            if np.nanargmax([p_g1, p_g2]) == 0:
                                in_tumor = 'in tumor'
                            else:
                                minor_allele_copy_number = segment_copy_number - minor_allele_copy_number
                                if minor_allele_copy_number == 0:
                                    in_tumor = 'not in tumor'
                        else:
                            sg_status = 'ambiguous_neither_G_nor_S'
                    else:
                        sg_status = 'unknown'
                    break
        zygosity = 'NA'
        if np.isnan(segment_copy_number) or np.isnan(minor_allele_copy_number) or purity < 0.19:
            zygosity = 'NA'
            if 'germline' in sg_status:
                in_tumor = 'NA'
        elif segment_copy_number == 0 and minor_allele_copy_number == 0:
            zygosity = 'homoDel'
        elif segment_copy_number == minor_allele_copy_number and minor_allele_copy_number == 1:
            zygosity = 'homozygous'
        elif segment_copy_number == minor_allele_copy_number and minor_allele_copy_number >= 2:
            zygosity = 'homozygous'
        elif segment_copy_number >= 1 and minor_allele_copy_number == 0:
            zygosity = 'not in tumor'
        elif segment_copy_number != minor_allele_copy_number and minor_allele_copy_number != 0:
            zygosity = 'het'

        # calculate allele burden (quantitative clonality)
        allele_burden = np.nan
        burden_ci = [np.nan, np.nan]
        if ('germline' in sg_status) or ('somatic' in sg_status):
            if 'germline' in sg_status:
                eaf = (purity * minor_allele_copy_number + 1 - purity) / (purity * segment_copy_number + 2 * (1 - purity))
            else:
                # elif 'somatic' in SG_status:
                eaf = purity * minor_allele_copy_number / (purity * segment_copy_number + 2 * (1 - purity))
            allele_burden = short_variant.frequency / eaf
            # sim 1000 times and get 95% CI of the allele burden
            # there was try/except
            eaf_sim = binom.rvs(int(short_variant.depth), eaf, size=1000) / short_variant.depth
            ab_sim = short_variant.frequency / eaf_sim  # take ratio
            burden_ci = np.percentile(ab_sim, [2.5, 97.5])
        results.append({
            'mutation': short_variant.mutation,
            'position': 'chr%s:%d' % (short_variant.chr_, short_variant.position),
            'depth': '%d' % short_variant.depth,
            'AF_obs': '%0.2f' % short_variant.frequency,
            'AF_E[G]': [af_g1, af_g2],
            'AF_E[S]': [af_s1, af_s2],
            'purity': purity,
            'CN': segment_copy_number,
            'M': minor_allele_copy_number,
            'errMAF': error_maf,
            'errLR': error_lr,
            'P(G)': max_prob_germline,
            'P(S)': max_prob_somatic,
            'log(G/S)': logodds,
            'call': sg_status,
            'zygosity': zygosity,
            'clonality': clonality,
            'in_tumor': in_tumor,
            'burden': allele_burden,
            'burdenCI': burden_ci,
        })
    return results


def read_cna_model_file(path):
    with open(path, 'r') as f:
        return [
            Segment(
                segment['CHR'], segment['segStart'], segment['segEnd'], segment['mafPred'], segment['CN'], segment['segLR'],
                segment['segMAF'], segment['numMAtumorPred'], segment['purity'], segment['baseLevel']
            )
            for segment in csv.DictReader(f, dialect='excel-tab')
        ]


def read_mut_aggr_full(path):
    with open(path, 'r') as f:
        return [
            ShortVariant(variant['mutation'], variant['frequency'], variant['depth'], variant['pos'])
            for variant in csv.DictReader(f, dialect='excel-tab')
        ]


class Segment:
    def __init__(self, chr_, start, end, maf_pred, copy_number, seg_lr, seg_maf, num_ma_tumor_pred, purity, base_level):
        super(Segment, self).__init__()
        self.chr_ = chr_.strip("chr")
        self.start = int(start)
        self.end = int(end)
        self.maf_pred = float(maf_pred) if maf_pred != 'NA' else np.nan

        self.copy_number = int(float(copy_number)) if copy_number != 'NA' else np.nan
        self.seg_lr = float(seg_lr) if seg_lr != 'NA' else np.nan
        self.seg_maf = float(seg_maf) if seg_maf != 'NA' else np.nan
        self.num_ma_tumor_pred = float(num_ma_tumor_pred) if num_ma_tumor_pred != 'NA' else np.nan
        self.purity = float(purity)
        self.base_level = float(base_level)


class ShortVariant:
    """
    lightweight structure for short variants
    """

    def __init__(self, mutation, frequency, depth, position):
        self.mutation = mutation
        self.frequency = float(frequency)
        self.depth = float(depth)
        self.chr_ = position.split(':')[0].strip('chr')
        self.position = int(position.split(':')[1])


def _arg_parser():
    script_version = globals().get('__version__')
    script_description = globals().get('__doc__')
    script_epilog = None
    script_usage = '''%(prog)s [options] aggregated_mutations_file cna_model_file
                 %(prog)s [-h|--help]
                 %(prog)s [--version]'''

    parser = argparse.ArgumentParser(usage=script_usage,
                                     description=script_description,
                                     epilog=script_epilog,
                                     add_help=False)

    g = parser.add_argument_group('Program Help')
    g.add_argument('-h', '--help', action='help',
                   help='show this help message and exit')
    g.add_argument('--version', action='version', version=script_version)

    g = parser.add_argument_group('Required Arguments')
    g.add_argument('aggregated_mutations_file', action='store', type=str,
                   help='Full aggregated mutations text file')
    g.add_argument('cna_model_file', action='store', type=str,
                   help='CNA model calls')

    g = parser.add_argument_group('Optional Arguments')
    g.add_argument('-o', dest='output_header', action='store', type=str, default='',
                   help='Output file folder and header')
    return parser


def main(args):
    np.random.seed(10)
    fname_muts = args.aggregated_mutations_file  # vars/sample11.mut_aggr.full.txt
    fname_cn_model = args.cna_model_file
    out_header = args.output_header
    if len(out_header) == 0:
        out_header = fname_muts[0:-4]
    sgz_full = out_header + '.fmi.sgz.full.txt'
    sgz_out = out_header + '.fmi.sgz.txt'
    # Sometimes we can not calculate a CNA model, so an empty file is supplied.  We need
    # to ensure that this script will produce its anticipated output file in this case.
    if not os.path.getsize(fname_cn_model):
        logger.warning("Empty cna_calls.txt file.  Not running SGZ code.")
        for filename in (sgz_full, sgz_out):
            open(filename, 'w').close()
        sys.exit(0)
    else:
        #############################################################
        # 1) Read CN model file
        cna_data = read_cna_model_file(fname_cn_model)
        # 2) Read mutations file
        short_variants = read_mut_aggr_full(fname_muts)
        # 3) Compute SGZ
        y_obj = sgz_core(cna_data, short_variants)
        # 4) Write output
        fout = open(sgz_full, 'w')
        fout2 = open(sgz_out, 'w')
        header = ['mutation', 'pos', 'depth', 'frequency', 'afG1', 'afS1', 'afG2', 'afS2',
                  'p', 'C', 'M', 'logOR_G', 'clonality', 'clonality_CI_low', 'clonality_CI_high',
                  'germline/somatic', 'zygosity']
        fout.write('\t'.join(header) + '\n')
        header = ['mutation', 'pos', 'depth', 'frequency', 'C', 'germline/somatic', 'zygosity']
        fout2.write('\t'.join(header) + '\n')

        for sv in y_obj:
            out_list = [
                sv['mutation'],
                sv['position'],
                sv['depth'],
                sv['AF_obs'],
                '%.2f' % sv['AF_E[G]'][0],
                '%.2f' % sv['AF_E[S]'][0],
                '%.2f' % sv['AF_E[G]'][1],
                '%.2f' % sv['AF_E[S]'][1],
                '%.3f' % sv['purity'],
                '%g' % sv['CN'],
                '%g' % sv['M'],
                '%.1f' % sv['log(G/S)'],
                '%.2f' % sv['burden'],
                '%.2f' % sv['burdenCI'][0],
                '%.2f' % sv['burdenCI'][1],
                sv['call'],
                sv['zygosity'],
            ]
            fout.write('\t'.join(out_list) + '\n')
            out_list = [
                sv['mutation'],
                sv['position'],
                sv['depth'],
                sv['AF_obs'],
                '%g' % sv['CN'],
                sv['call'],
                sv['zygosity'],
            ]
            fout2.write('\t'.join(out_list) + '\n')
        fout.close()
        fout2.close()


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: [%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Here, we collect the arguments into a dictionary keyed on argname.
    parsed_args = _arg_parser().parse_args(sys.argv[1:])
    if 'debug' in parsed_args:
        logger.setLevel(logging.DEBUG)
    sys.exit(main(parsed_args))
