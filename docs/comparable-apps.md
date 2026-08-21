# Comparable apps and product patterns

Reviewed 2026-08-20 against current first-party documentation.

## trainerd's product boundary

`trainerd` is a small, self-hosted control plane for queued subprocess work. A
client chooses an allowlisted project or repository task and bounded parameters;
the daemon owns commands, paths, concurrency, checkout state, logs, cancellation,
validation, and promotion. That puts it between an operations runbook server and
a self-hosted CI or data-workflow worker. It is not an experiment tracker or a
distributed scheduler. See the [trainerd README](../README.md).

## APOSD design audit

The scan covered all 34 tracked files. Scores use the 10 production Python
modules, 2,997 lines, as the measured interface. Tests, examples, and automation
files were read to confirm contracts and repeated policy.

| Dimension | Score | Count and evidence |
|---|---:|---|
| Pass-through proliferation | 3/4 | One wrapper, `_resolve_cmd`, repeats `_resolve_template`'s signature and delegates directly. [`runner.py:283`](../src/trainerd/runner.py#L283) |
| Information duplication | 0/4 | 21 distinct contract or policy decisions recur across independent modules. The largest cluster is the same 11 request fields in [`contracts.py:17`](../src/trainerd/contracts.py#L17) and [`server.py:86`](../src/trainerd/server.py#L86). |
| Interface documentation | 2/4 | 17 of 42 public functions and methods have interface docstrings, 40.5 percent. Nine of 12 `JobStore` methods are undocumented. [`storage.py:58`](../src/trainerd/storage.py#L58) |
| Naming quality | 2/4 | Three blocklisted local names: `handle`, `tmp`, and `data`. [`managed_env.py:100`](../src/trainerd/managed_env.py#L100) [`runner.py:442`](../src/trainerd/runner.py#L442) [`storage.py:82`](../src/trainerd/storage.py#L82) |
| Exception discipline | 0/4 | Three custom exception types and six catch blocks whose body is only `pass`. Examples: schema migration and JSON decoding in [`storage.py:53`](../src/trainerd/storage.py#L53), permission changes in [`managed_env.py:110`](../src/trainerd/managed_env.py#L110), and validation JSON in [`runner.py:465`](../src/trainerd/runner.py#L465). |
| **Total** | **7/20, Poor** | Four dimensions score 2 or lower, so tactical-tornado risk is high under the audit rubric. |

The 21 duplicated decisions are 11 job-request fields, one safe-identifier rule,
one environment-name rule, one step-timeout default, three validation defaults,
two path-template placeholders, one Git checkout-update policy, and one terminal
job-status set. The safe identifier appears in
[`config.py:91`](../src/trainerd/config.py#L91),
[`lan.py:28`](../src/trainerd/lan.py#L28), and
[`server.py:99`](../src/trainerd/server.py#L99). Validation defaults appear in
[`config.py:35`](../src/trainerd/config.py#L35) and
[`lan.py:330`](../src/trainerd/lan.py#L330).

Five findings passed the audit's file, line, pattern, count, and action gate:

- **P1, duplicated request and policy contracts, 21 decisions.** Move
  `JobRequest` into the contract module and derive its JSON schema from that one
  model. Reuse one safe-identifier validator and one terminal-status set.
- **P2, undocumented public interfaces, 25 of 42.** Document the caller contract
  for `JobStore`, `JobRunner`, CLI entry points, and HTTP handlers. Start with
  lifecycle and persistence methods, where callers otherwise must read SQL or
  subprocess code.
- **P2, swallowed exceptions, six blocks, plus three custom exceptions.** Replace
  broad SQLite migration and JSON parse suppression with checks for the exact
  expected condition. Keep the three boundary-specific exception types.
- **P2, vague local names, three identifiers.** Rename them to `env_file`,
  `output_file`, and `job` at the cited locations.
- **P3, pass-through command resolver, one function.** Delete `_resolve_cmd` and
  call `_resolve_template` directly.

The design has strong boundaries despite the score. LAN requests cannot choose
commands or host paths. Repository sync pins each run to a logged revision.
`managed_env` writes atomically and injects only declared names. The test suite
also enforces domain-neutral source code.

The next structural change should remove state decisions from HTTP handlers.
Splitting `server.py` into route, service, and repository layers would add
pass-throughs. A deeper `TrainerdRuntime` object is the better option: it can own
projects, queue reservations, LAN locks, and job lookup behind a small interface,
replacing the module globals at [`server.py:49`](../src/trainerd/server.py#L49).

## Comparable products and use cases

| Product | Main use case | Relevant overlap and difference |
|---|---|---|
| [Rundeck](https://docs.rundeck.com/docs/manual/jobs/index.html) | Self-service operational procedures and runbooks | Jobs encapsulate options, steps, node selection, error behavior, and access policy. Rundeck stores executions, exposes step output, and permits aborts. It is the closest match for trainerd registry mode, but adds users, node orchestration, schedules, retries, and notifications. [Job execution](https://docs.rundeck.com/docs/manual/jobs/) [Retries and schedules](https://docs.rundeck.com/docs/manual/jobs/creating-jobs.html) [Notifications](https://docs.rundeck.com/docs/manual/jobs/job-notifications.html) |
| [Ray Jobs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/rest.html) | Submit and manage applications on a Ray cluster | Its HTTP API submits work asynchronously and supports list, status, logs, and stop operations. Ray also accepts an entrypoint, working directory, runtime environment, and CPU or GPU reservations. Trainerd deliberately narrows that interface to named tasks and server-controlled commands. [Runtime environments and resources](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html) |
| [Buildkite Agent](https://buildkite.com/docs/agent) | Run CI and deployment pipelines on private machines | The agent checks out source before a command step; repository YAML defines work. Queues route work to machine classes, concurrency groups prevent collisions, and jobs support retries and artifacts. This is the closest match for trainerd LAN mode, although Buildkite supplies a hosted control plane. [Checkout](https://buildkite.com/docs/pipelines/configure/git-checkout) [Pipeline steps](https://buildkite.com/docs/pipelines/configure/defining-steps) [Queues](https://buildkite.com/docs/agent/queues) [Concurrency](https://buildkite.com/docs/pipelines/configure/workflows/controlling-concurrency) [Retries](https://buildkite.com/docs/pipelines/configure/retry) [Artifacts](https://buildkite.com/docs/agent/v3/cli/reference/artifact) |
| [Prefect](https://docs.prefect.io/v3/concepts/deployments) | Schedule and remotely run data workflows | Deployments bind parameters, code, infrastructure, schedules, and concurrency policy. Workers poll a work pool and can execute flows as local subprocesses. Prefect fits recurring training and ETL work, but its workflow graph and infrastructure adapters exceed trainerd's one-host scope. [Workers](https://docs.prefect.io/v3/concepts/workers) |
| [MLflow Projects](https://mlflow.org/docs/latest/ml/projects/) | Package reproducible ML programs and run named entry points | An `MLproject` file declares entry points, typed parameters, commands, and environments; a project can come from a Git URI and run locally or through a remote backend. MLflow Tracking records parameters, code versions, metrics, and artifacts. It complements trainerd's execution and promotion flow rather than replacing its queue. [Tracking](https://mlflow.org/docs/latest/ml/tracking/) |

## Common functionality

The products converge on a small core that trainerd already has:

- A named, declarative unit of work rather than raw remote shell access.
- An asynchronous lifecycle with stable job IDs, status, logs, and cancellation.
- Queue and concurrency controls that prevent jobs from colliding on shared
  hosts, repositories, or deployment targets.
- A fixed code revision or managed checkout attached to each run.
- Server-side secret injection, bounded client parameters, and persistent run
  records.

The distinctions are useful. Rundeck owns operator policy, Buildkite owns CI
orchestration, Prefect owns scheduled data workflows, Ray owns distributed
resources, and MLflow owns experiment records. Copying all five would turn
trainerd into a shallow version of each.

## Improvement candidates

The issue tracker now records this order:

- **P0:** Kill the full Windows subprocess tree on cancellation, [issue
  25](https://github.com/Kh1ng/trainerd/issues/25). Stream newline-free child
  output in bounded chunks, [issue
  22](https://github.com/Kh1ng/trainerd/issues/22).
- **P1:** Make Windows upgrades reversible, [issue
  18](https://github.com/Kh1ng/trainerd/issues/18). Add bounded artifact
  retrieval, [issue 19](https://github.com/Kh1ng/trainerd/issues/19). Allow LAN
  clients to choose declared steps, [issue
  23](https://github.com/Kh1ng/trainerd/issues/23).
- **P2:** Persist LAN allowlists, [issue
  20](https://github.com/Kh1ng/trainerd/issues/20). Add client profiles, [issue
  24](https://github.com/Kh1ng/trainerd/issues/24). Consolidate contracts and
  validation policy, [issue 26](https://github.com/Kh1ng/trainerd/issues/26).
  Move daemon state behind one interface, [issue
  27](https://github.com/Kh1ng/trainerd/issues/27).

After those items, the comparable products suggest these conditional additions:

1. **Retry attempts with lineage.** Rundeck and Buildkite retain retry behavior
   and history. Add a bounded retry policy only after transient failures become
   common; keep every attempt attached to the original job. [Rundeck retry](https://docs.rundeck.com/docs/manual/jobs/creating-jobs.html#retry) [Buildkite retry](https://buildkite.com/docs/pipelines/configure/retry)
2. **Schedules or signed completion webhooks.** Prefect supports scheduled and
   event-triggered deployments, while Rundeck sends notifications on start,
   success, failure, and retryable failure. Add one of these only if external
   cron and polling become an operating burden. [Prefect deployments](https://docs.prefect.io/v3/concepts/deployments) [Rundeck notifications](https://docs.rundeck.com/docs/manual/jobs/job-notifications.html)
3. **Resource-aware routing.** Buildkite queues target operating systems and
   machine classes; Ray reserves CPU, GPU, memory, and custom resources. Add
   worker labels or resource requirements when trainerd grows beyond one
   interchangeable host. [Buildkite queues](https://buildkite.com/docs/agent/v3/queues/managing) [Ray resource reservations](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#specifying-cpu-and-gpu-resources)
4. **Run metadata and external artifact storage.** MLflow records parameters,
   metrics, code versions, and output files; Buildkite can upload artifacts to
   its store or private object storage. Prefer emitting an MLflow-compatible run
   reference or object-store URI over building another experiment tracker.
   [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) [Buildkite artifacts](https://buildkite.com/docs/agent/v3/cli/reference/artifact)
5. **Isolation and narrower authorization.** GitHub recommends one-job
   ephemeral self-hosted runners to prevent state and secrets from crossing job
   boundaries. Rundeck applies separate privileges to read, create, edit, run,
   and kill jobs. Add per-run isolation when repositories stop being mutually
   trusted; add scoped identities when one API key no longer maps to one trusted
   operator group. [GitHub ephemeral runners](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#ephemeral-runners-for-autoscaling) [Rundeck job access](https://docs.rundeck.com/docs/manual/jobs/index.html)

## Design recommendation

Keep trainerd's interface centered on `submit named task -> observe job ->
validate -> promote`. Retries fit inside that lifecycle. Scheduling, metrics,
artifact storage, and identity systems already have mature owners and should be
integration points unless a concrete deployment cannot use those tools.

## Primary sources

- [Rundeck job documentation](https://docs.rundeck.com/docs/manual/jobs/index.html)
- [Ray Jobs REST API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/rest.html)
- [Buildkite agent and pipeline documentation](https://buildkite.com/docs/agent)
- [Prefect deployments](https://docs.prefect.io/v3/concepts/deployments)
- [MLflow Projects](https://mlflow.org/docs/latest/ml/projects/)
- [GitHub self-hosted runner reference](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
