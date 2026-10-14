# HEP scheduling simulator

An event-driven Python simulation of a distributed computing grid. Jobs form DAG workflows and compete for CPU cores, disk storage and network bandwidth. Policies choose the site for each job; the simulator handles transfers, execution and file eviction.

The repository includes heuristic policies and a basic model-driven agent that can read the simulator, write a scheduler, run simulations and inspect results. Research search strategies, private skill libraries, training pipelines and recorded results are excluded.

## Quick start

Use Python 3.10 or newer. Run commands from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example.py
python baselines.py --config environment
```

`environment` is an explicit grid configuration. `small_scale`, `middle_scale` and `large_scale` generate grids from seeded distributions. YAML files control sites, workloads, files, storage and network settings. Generated configurations use very large disk capacities; reduce these to explore eviction behavior.

## Custom scheduling policy

Save a policy as `schedulers/my_scheduler.py`:

```python
from env import SchedulerInterface

class MyScheduler(SchedulerInterface):
    def schedule(self, job, env):
        return max(range(len(env.sites)),
                   key=lambda i: env.sites[i].num_free_cpus)
```

Run it with `python baselines.py --config environment --schedulers schedulers/my_scheduler.py MCT`.

See [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md) for the scheduler interface and available state. Policies must read only arrived jobs and must not mutate the environment. `sim.summary()`, `sim.job_stats()` and `sim.site_stats()` expose simulation results. The agent objective, `throughput_charged_gflops_s`, is completed work divided by summed job spans, including elapsed waits for unfinished submitted jobs; it is not aggregate work per wall-clock second.

## Basic agent

```bash
pip install -r requirements-agent.txt
export ANTHROPIC_API_KEY="your-key"
python run_agent.py --backend anthropic --model YOUR_MODEL --max-turns 10
```

For a local or hosted OpenAI-compatible endpoint:

```bash
# Set OPENAI_API_KEY if the endpoint requires authentication.
python run_agent.py --backend openai --base-url http://localhost:8000/v1 --model YOUR_MODEL
```

The retained prompt is in `agent/loop.py`; four tool schemas and their implementation are in `agent/tools.py`. The generated policy goes to `schedulers/agent_scheduler.py`. Logs and diagnostic snippets go to `analysis/`. These outputs are ignored by Git. Agent runs can incur API charges and execute model-generated Python with the permissions of your process. Use a disposable environment with no sensitive files for untrusted models or policies; the read restrictions are not a security sandbox.

## Layout

- `env/`: simulation, job and file models, network transfers and eviction.
- `schedulers/`: MinT, MinE, MCT, DIANA, least-loaded and random policies.
- `agent/`: basic conversation loop, model backends and simulation tools.
- `config/`: example simulation configurations.
- `example.py`: simulator-only entry point.
- `baselines.py`: policy comparison command.
- `run_agent.py`: optional model-driven agent entry point.

Before publishing, choose a license and review the files you intend to commit. Do not commit API keys or generated run artifacts.
