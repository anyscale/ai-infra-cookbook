# Contributing an example

Every example in this repository is a self-contained directory with code, a `Dockerfile` (when dependencies go beyond the base image), an Anyscale `job.yaml` or `service.yaml`, and a README that follows the shared template below. The example also gets an entry in the [catalog](site/src/data/catalog.ts) that places it on the [map](README.md#the-map).

## Checklist

1. **Create the example directory** — code, `job.yaml`/`service.yaml`, `Dockerfile`, `README.md`.
2. **Write the README from the template** below. Preserve the section order; the consistency checker is lenient about extra sections but strict about links and commands.
3. **Add a catalog entry** in [`site/src/data/catalog.ts`](site/src/data/catalog.ts): stage, summary, stack with citation URLs, run commands (must appear *verbatim* in your README), key files, and edges to upstream/downstream examples. If the example should appear on [docs.anyscale.com/tutorials](https://docs.anyscale.com/tutorials/), also give it a `docs: { slug, description }` mapping.
4. **Update the mermaid map** in [README.md](README.md#the-map) if your example adds a node or edge.
5. **Add your example to the stage table** in the root README, and to a journey in the catalog if it extends one.
6. **Run the checks** — they validate that the catalog, the READMEs, and the map agree:

   ```bash
   cd site
   npm install
   npm run check
   ```

   The checker enforces: every directory has a catalog entry (and vice versa), edges and journeys reference real examples, every `blob/main` link points at a file that exists, every catalog command appears verbatim in its README, each README's "Position in the stack" links match the catalog edges, and the root README mentions every example.

7. **Sync the docs mirror.** READMEs are republished as tutorials on [docs.anyscale.com/tutorials](https://docs.anyscale.com/tutorials/). After README changes merge, regenerate the mirror into a checkout of [anyscale/docs](https://github.com/anyscale/docs) and open a PR there:

   ```bash
   cd site
   npm run export:docs -- /path/to/anyscale-docs
   ```

   The exporter rewrites sibling links to `/tutorials/<slug>.md`, points root-README links at GitHub, moves images into `static/img/tutorials/<slug>/`, and regenerates the stage-grouped tutorials index.

## README template

````markdown
# {Capability-first title, e.g. "GRPO post-training with SkyRL"}

{1–3 sentences: what this example does and which open-source libraries it
showcases, with inline links to each library's repository.}

## The stack

| Layer | Library | Role in this example |
|---|---|---|
| {e.g. Training} | [{Library}]({upstream repo URL}) | {what it does in this example} |
| Orchestration | [Ray {Data,Train,Serve}](https://docs.ray.io/...) | {role} |
| Platform | [Anyscale](https://www.anyscale.com) | image build, compute provisioning, job/service management |

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job        <!-- or "Deploy the service" -->

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/{dir}
```

{Submit/deploy command, including any --env flags the example needs.}

```bash
anyscale job submit -f job.yaml
```

## Understanding the example

- {Key technical points, linking files with the absolute convention:
  https://github.com/anyscale/examples/blob/main/{dir}/{file}}

{Example-specific sections: ## Query the service / ## View the results / ...}

{## Shutdown — services only}

## Position in the stack

**Stage:** {Curate | Train | Post-train | Serve | Foundations}

- **Upstream:** [{title}](../{dir}/) — {what flows in}
- **Downstream:** [{title}](../{dir}/) — {what flows out}
- **Related:** [{title}](../{dir}/) — {alternative approach}
- **Journeys:** [{Journey title}](../README.md#journeys)

Part of the [Open-Source Frontier Infra Stack](../README.md) — explore the
map in the [interactive explorer](../README.md#interactive-explorer).
````

Conventions:

- **Navigation links between examples are relative** (`../{dir}/`) so they work on GitHub and in local checkouts. **File deep-links are absolute** (`blob/main`) — the checker validates both.
- Omit empty Position groups (an example with no upstream simply has no Upstream line).
- Tutorial-style examples (`job_hello_world`, `service_hello_world`) may use numbered step headings; library-style examples (`lerobot_datasource`) skip the Anyscale CLI sections. Keep the stack table and Position section in all variants.
- Cite the canonical upstream repository for every library the example showcases — the [Built on open source](README.md#built-on-open-source) table in the root README lists the roster.
