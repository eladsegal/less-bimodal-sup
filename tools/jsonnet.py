import argparse
import sys, os
import json
from _jsonnet import evaluate_snippet
import string
import random
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.config import handle_config, TEMPLATES_DIR
from utils.jsonnet import evaluate
from utils.general import resolve_relative_paths


random_temp_name = (
    f"temp_{''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(5))}"
)


@hydra.main(config_path="../configs/templates", config_name=random_temp_name)
def main(cfg: DictConfig):
    os.remove(resolve_relative_paths(os.path.join(TEMPLATES_DIR, f"{random_temp_name}.yaml")))

    print(json.dumps(OmegaConf.to_container(cfg), indent=4, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("path", type=str, help=".jsonnet/.ckpt path")
    parser.add_argument("--is_eval", action="store_true", default=False)
    parser.add_argument("--no_mandatory_verification", action="store_true", default=False)
    parser.add_argument(
        "--simple",
        action="store_true",
        default=False,
        help="Don't use the config handler, just output the jsonnet as is",
    )
    args, unknown = parser.parse_known_args()

    if args.simple:
        config = evaluate(args.path)
        print(json.dumps(config, indent=4, sort_keys=True))
    else:
        excluded_args = ["path", "is_eval", "no_mandatory_verification", "simple"]
        sys.argv = (
            [sys.argv[0]]
            + [args.path]
            + [f"{k}={v}" for k, v in vars(args).items() if k not in excluded_args]
            + unknown
        )
        handle_config(
            random_temp_name,
            is_silent=False,
            is_eval=args.is_eval,
            mandatory_verification=not args.no_mandatory_verification,
        )
        main()
