# flux-sandbox

> Safe simulation environment for testing cooperative FLUX programs

Before deploying cooperative execution to the real fleet, we need a sandbox where agent interactions can be tested safely — mock agents, mock repos, deterministic execution, and no risk of corrupting fleet state.

## Capabilities

1. **Mock Agents** — Simulated fleet agents with configurable behaviors
2. **Mock Repos** — Local git repos that mirror fleet structure
3. **Deterministic Execution** — Same inputs always produce same outputs
4. **Failure Injection** — Simulate network failures, agent crashes, bad results
5. **Cooperation Scenarios** — Pre-built test scenarios for common cooperation patterns

## Relationship to flux-coop-runtime

flux-coop-runtime's tests use this sandbox. Phase 1 spec tests require: one ask, one respond, one timeout. Phase 2 tests require: delegate to 3 agents, 1 fails, 2 succeed.

## Status

Schema pushed. Awaiting mock agent implementations and test scenarios.

---

*"Break it here so it doesn't break out there."*
