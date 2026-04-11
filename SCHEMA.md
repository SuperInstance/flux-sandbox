# flux-sandbox Schema

```
flux-sandbox/
├── README.md
├── SCHEMA.md
├── src/
│   ├── mocks/
│   │   ├── agent.py           # Mock fleet agent
│   │   ├── repo.py            # Mock git repository
│   │   ├── router.py          # Mock semantic router
│   │   └── transport.py       # Mock message transport
│   ├── scenarios/
│   │   ├── phase1_ask.py      # Phase 1 test scenarios
│   │   ├── phase2_delegate.py # Phase 2 test scenarios
│   │   └── phase3_coiter.py   # Phase 3 test scenarios
│   ├── harness/
│   │   ├── runner.py          # Deterministic test runner
│   │   ├── recorder.py        # Record and replay executions
│   │   └── inspector.py       # Inspect execution traces
│   ├── failure/
│   │   ├── injector.py        # Inject failures
│   │   └── presets.py         # Common failure scenarios
│   └── tests/
├── fixtures/                  # Static test data
└── message-in-a-bottle/
    └── for-fleet/
```
