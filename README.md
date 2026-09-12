# chia-braggnn-workspace

いろいろと面倒なので、すべての操作（ジョブ投入含む）は `platinum.perf.cs.tut.ac.jp` 上で行ってください。

## 事前準備

事前に uv (https://docs.astral.sh/uv/) をインストールしておいてください。

ホームディレクトリ直下などにcloneします。

```bash
git clone https://github.com/ANL-TUT/chia-braggnn-workspace.git
cd chia-braggnn-workspace
uv sync
```

おしまい

## ジョブ投入

```bash
uv run chia job submit --address http://133.15.45.28:8265 --working-dir . -- python chia/braggnn_loop.py --iterations 5
```

## システム構成

LLM (OpenCode + Gemini) で `exo/braggnn_exo.py` を最適化するループ

```mermaid
graph TD
    subgraph head["head (133.15.45.28)"]
        D["chia/braggnn_loop.py<br/>ray job driver"]
    end

    subgraph llm["opencode worker — docker: chia-opencode<br/>resources: opencode_creds 1"]
        L["OpenCodeLLM.prompt<br/>Gemini on Vertex AI"]
    end

    subgraph exo["exo_compiler worker — docker: chia-exo<br/>resources: exo_build 8"]
        B["exo_bash (BashTool)<br/>edit braggnn_exo.py"]
        E["build_elf<br/>exocc + riscv64 gcc (htif baremetal)"]
    end

    subgraph fire["perflab_firesim (133.15.45.113)<br/>resources: firesim 1"]
        F["run_workload<br/>Alveo U250 / Rocket + Gemmini"]
    end

    D -->|prompt| L
    L -->|MCP tool calls| B
    D -->|work_dir| E
    E -->|braggnn.riscv| D
    D -->|ELF| F
    F -->|"uartlog: cycles, PASSED / FAILED"| D
```

## クラスタ立ち上げ（やらないでください）

`chia up` `chia down` は物理マシンに対して複数人がやるとめちゃくちゃになってしまうとおもうので、やらないでください。

```bash
./scripts/deploy.sh
```
