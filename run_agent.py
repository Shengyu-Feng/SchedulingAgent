#!/usr/bin/env python3
"""Run the basic scheduling agent using a hosted or local model."""
import argparse
import json

from agent import AnthropicBackend, OpenAIBackend, run_research_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['anthropic', 'openai'], default='anthropic')
    parser.add_argument('--model', required=True, help='Model identifier accepted by your endpoint')
    parser.add_argument('--base-url', default='http://localhost:8000/v1')
    parser.add_argument('--max-turns', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if args.max_turns < 1:
        parser.error('--max-turns must be positive')
    if args.backend == 'anthropic':
        backend = AnthropicBackend(model=args.model)
    else:
        backend = OpenAIBackend(base_url=args.base_url, model=args.model,
                                temperature=args.temperature)
    results = run_research_agent(backend, max_turns=args.max_turns, verbose=not args.quiet)
    print(json.dumps({f'{scheduler}/{config}': data['summary']
                      for (scheduler, config), data in results.items()}, indent=2))


if __name__ == '__main__':
    main()
