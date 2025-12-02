import logging
import math
import sys
from datetime import timedelta

from train_dqn import parse_args, train

DEFAULT_TOTAL_STEPS = 200_000
DEFAULT_RENDER_MODE = "human"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    args = parse_args()
    if args.render is None:
        args.render = True
    if args.render and not args.render_mode:
        args.render_mode = DEFAULT_RENDER_MODE
    args.total_steps = DEFAULT_TOTAL_STEPS if args.total_steps == 1_000_000 else args.total_steps
    configure_logging()
    logger = logging.getLogger("main")

    logger.info("Arguments: %s", vars(args))
    metrics = train(args)

    elapsed = timedelta(seconds=metrics["total_time"])
    best_mean = metrics["best_mean_return"]
    if not math.isfinite(best_mean):
        best_mean_display = float("nan")
    else:
        best_mean_display = best_mean
    logger.info(
        "Training summary | episodes=%d | steps=%d | best_mean_return=%.2f | time=%s | steps/s=%.1f | tensorboard=%s",
        metrics["episodes"],
        metrics["steps"],
        best_mean_display,
        elapsed,
        metrics["steps_per_second"],
        metrics["tensorboard_run"],
    )


if __name__ == "__main__":
    main()
