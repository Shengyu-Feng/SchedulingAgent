"""Run a simulation without an API key or model server."""
import argparse
import json
from pathlib import Path

from env import HEPSimulator
from schedulers import MCTScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='environment',
                        choices=['environment', 'small_scale', 'middle_scale', 'large_scale'])
    parser.add_argument('--until', type=float, default=3600.0)
    args = parser.parse_args()
    config = Path(__file__).parent / 'config' / f'{args.config}.yaml'
    sim = HEPSimulator.from_yaml(str(config), scheduler=MCTScheduler())
    sim.run(until=args.until)
    print(json.dumps(sim.summary(), indent=2))


if __name__ == '__main__':
    main()
