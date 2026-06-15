from __future__ import annotations

from .pipeline import build_arg_parser, config_from_args, run_pipeline


def main() -> int:
    parser = build_arg_parser()
    cfg = config_from_args(parser.parse_args())
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

