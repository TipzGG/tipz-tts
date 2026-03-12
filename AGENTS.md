## Session Notes

- For XTTS inference with the `beerschool` voice on GPU, never run multiple `cli.py infer` processes in parallel. Loading the same checkpoint concurrently can exhaust VRAM and freeze the machine.
- When comparing `temperature` values, run them serially, one process at a time.
- Prefer validating with a single inference first after a reboot or environment change.
