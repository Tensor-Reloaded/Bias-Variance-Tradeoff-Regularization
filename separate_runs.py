import yaml
import shutil
import os
import sys

SRC_ROOT = r'multirun'
DST_ROOT = r'separated_runs'


def main():
    for root, dirs, files in os.walk(SRC_ROOT):
        if root.endswith('.hydra'):
            assert 'config.yaml' in files
            conf = os.path.join(root, 'config.yaml')
            d = os.path.join(root, '..')
            d = os.path.abspath(d)
            d = os.path.relpath(d, os.path.curdir)
            with open(conf, 'r') as f:
                y = yaml.safe_load(f)
                dst_name = d.replace('\\', '_')
                if dst_name.startswith('multirun_2020-'):
                    dst_name = dst_name[len('multirun_2020-'):]

                def append_if_exists(long_name, short_name):
                    nonlocal y
                    nonlocal dst_name
                    val = y.get(long_name, "")
                    if val:
                        dst_name += f'_{short_name}={val}'
                    return val

                lips = append_if_exists("lipschitz_regularization", "lips")
                homo = append_if_exists("homomorphic_regularization", "homo")
                if not (homo or lips):
                    dst_name += f'_baseline'

                append_if_exists("homomorphic_regularization_factor", "hf")

                if homo or lips:
                    append_if_exists("level", "lvl")

                append_if_exists("homomorphic_level", "hlvl")
                append_if_exists("lipschitz_level", "llvl")

                append_if_exists("lipschitz_regularization_loss_factor", "llf")
                append_if_exists("lipschitz_noise_factor", "lnf")
                append_if_exists("lipschitz_noise_factor_gamma", "lnfg")

                subset = y.get("train_subset", None)
                dst_name += f'_subset={subset}'

                if not os.path.exists(os.path.join(DST_ROOT, dst_name)):
                    print(os.path.join(DST_ROOT, dst_name))
                # print(dst_name)
                # shutil.copytree(src=d, dst=os.path.join(DST_ROOT, dst_name))

if __name__ == '__main__':
    main()