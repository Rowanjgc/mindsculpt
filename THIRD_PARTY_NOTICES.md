# Third-party notices

WSP Workbench is licensed under Apache-2.0. No third-party source code is
bundled in this repository. The adapters in `wsp/adapters/` are independent
implementations of documented data formats.

The following projects were reviewed as prior art or interoperability targets.
Their licenses are recorded for attribution and planning; they are not
dependencies of the core package unless a user installs an optional extra.

| Project | License | Use in this repository |
| --- | --- | --- |
| Inspect AI (`UKGovernmentBEIS/inspect_ai`) | MIT | Log format interoperability (`wsp.adapters.inspect_eval`); not imported |
| Inspect Scout (`meridianlabs-ai/inspect_scout`) | MIT | Finding-vocabulary reference; not a dependency |
| LangTest (`PacificAI/langtest`) | Apache-2.0 | Perturbation reference; not a dependency |
| INIF (`ndif-team/inif`) | MIT | Format 0.1 interoperability (`wsp.adapters.inif`); package not imported |
| NNsight (`ndif-team/nnsight`) | MIT | Feasibility review only; no code or dependency |
| nnterp (`ndif-team/nnterp`) | MIT | Feasibility review only; no code or dependency |
| TransformerLens | MIT | Reference only |
| pyvene | Apache-2.0 | Reference only |
| LM Evaluation Harness (`EleutherAI/lm-evaluation-harness`) | MIT | Reference only |
| EasyEdit (`zjunlp/EasyEdit`) | MIT | Reference only |
| SAELens (`decoderesearch/SAELens`) | MIT | Reference only |

Optional runtime dependencies keep their own licenses:

- PyTorch (BSD-3-Clause)
- NumPy (BSD-3-Clause)
- SafeTensors (Apache-2.0)
- pydantic (MIT)
- Typer (MIT)
- Transformers, when installed via the `hf` extra (Apache-2.0)

See `docs/prior-art-and-integration.md` for versions, tested configurations, and
the reasons each integration was adopted, adapted, deferred, or only referenced.
