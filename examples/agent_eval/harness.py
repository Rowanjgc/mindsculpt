"""Autonomous LLM agent driving the WSP MCP server over stdio.

The agent model (hosted, OpenAI-compatible) decides every tool call; the
harness only:
  * connects to `wsp mcp` via the MCP SDK client,
  * passes the server's own MCP instructions to the model as system prompt,
  * renders MCP tools as OpenAI function definitions (full surface by
    default, or progressive disclosure via --progressive),
  * records a full transcript with per-step usage.

Run with the project venv: .venv/bin/python harness.py ...
API key is read from OPENCODE_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ZEN = "https://opencode.ai/zen/v1"
MAX_ENVELOPE_CHARS = 12000
# Repo root: the parent of this file's directory (examples/agent_eval/).
REPO = Path(__file__).resolve().parent.parent.parent

FRAMING = (
    "You are operating a scientific instrument through MCP tools. "
    "Every tool returns a wsp.tool/1.0 JSON envelope: ok=true means success, "
    "ok=false means a structured error in error.code / error.message / "
    "error.details that you should read and act on; a corrected retry is "
    "allowed and expected when the error is actionable. Artifact ids returned "
    "by one tool are the inputs of later tools. Never invent operations or "
    "payload fields; use only what the schemas, instructions, and error "
    "messages tell you. When you have gathered enough evidence, reply with a "
    "final answer (no tool call) that reports the requested ids and findings."
)


def api_chat(payload: dict, key: str, *, retries: int = 3) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            ZEN + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Default urllib User-Agent is blocked by the edge (403).
                "User-Agent": "wsp-agent-eval/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"API failed after {retries} attempts: {last!r}")


def render_tool(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def discover_operation_names(envelope: dict) -> list[str]:
    """Operation names advertised by a discover_capabilities response."""
    data = envelope.get("data") or {}
    names: list[str] = []
    for source in ("next_operations", "operations"):
        for op in data.get(source) or []:
            if isinstance(op, dict):
                name = op.get("name") or op.get("operation")
                if name:
                    names.append(name)
    seen, ordered = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


class AgentRun:
    def __init__(self, *, key: str, model: str, workspace: Path, task: str,
                 max_steps: int, progressive: bool, log=print):
        self.key = key
        self.model = model
        self.workspace = workspace
        self.task = task
        self.max_steps = max_steps
        self.progressive = progressive
        self.log = log
        self.messages: list[dict] = []
        self.tools: dict[str, dict] = {}
        self.session: ClientSession | None = None
        self._stdio = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        self.errors = []  # (step, tool, code) for ok:false envelopes

    # -- MCP ----------------------------------------------------------------
    async def connect(self) -> None:
        params = StdioServerParameters(
            command=str(REPO / ".venv" / "bin" / "wsp"),
            args=["mcp", "--workspace", str(self.workspace), "--auto-init"],
            cwd=str(REPO),
        )
        self._stdio = stdio_client(params)
        read, write = await self._stdio.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        result = await self.session.initialize()
        self.server_info = {
            "name": result.server_info.name,
            "version": result.server_info.version,
            "instructions": result.instructions or "",
        }
        listing = await self.session.list_tools()
        self._all_tools = {t.name: t for t in listing.tools}
        if self.progressive:
            for name in ("discover_capabilities", "inspect_model", "workspace_status"):
                self.tools[name] = self._all_tools[name]
        else:
            self.tools = dict(self._all_tools)
        self.log(f"mcp connected; server={self.server_info['name']} "
                 f"v{self.server_info['version']}; exposed {len(self.tools)} tools")

    async def close(self) -> None:
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._stdio:
            await self._stdio.__aexit__(None, None, None)

    async def call(self, name: str, arguments: dict) -> dict:
        result = await self.session.call_tool(name, arguments)
        structured = result.structured_content
        if not structured:
            for item in result.content or []:
                if getattr(item, "text", None):
                    structured = json.loads(item.text)
                    break
        return structured or {}

    def widen(self, envelope: dict) -> int:
        added = 0
        for name in discover_operation_names(envelope):
            if name not in self.tools and name in self._all_tools:
                self.tools[name] = self._all_tools[name]
                added += 1
        return added

    # -- model loop -----------------------------------------------------------
    def chat(self) -> tuple[dict, dict]:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "tools": [render_tool(t) for t in sorted(self.tools.values(), key=lambda t: t.name)],
            "temperature": 0.1,
            # No max_tokens cap: glm-5.3 spends tokens on reasoning_content
            # before emitting content/tool_calls, and an artificial cap just
            # starves it mid-thought. Let the API default apply.
        }
        resp = api_chat(payload, self.key)
        choice = resp["choices"][0]
        usage = resp.get("usage") or {}
        cost = ((usage.get("opencode") or {}).get("list_cost")) or 0.0
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.usage["cost"] += cost
        message, finish = choice["message"], choice.get("finish_reason")
        # Empty responses are usually glm reasoning tokens exhausting
        # max_tokens (finish=length, content=null, reasoning_content set).
        # Retry a few times before giving up.
        for _ in range(2):
            if message.get("tool_calls") or message.get("content"):
                break
            self.log(f"  !! empty assistant message: finish={finish} "
                     f"keys={sorted(message.keys())} "
                     f"raw={json.dumps(choice)[:400]}")
            resp = api_chat(payload, self.key)
            choice = resp["choices"][0]
            usage = resp.get("usage") or {}
            cost = ((usage.get("opencode") or {}).get("list_cost")) or 0.0
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.usage["cost"] += cost
            message, finish = choice["message"], choice.get("finish_reason")
        return message, finish

    def truncate(self, envelope: dict) -> str:
        body = json.dumps(envelope)
        if len(body) > MAX_ENVELOPE_CHARS:
            body = body[:MAX_ENVELOPE_CHARS] + " ...[truncated]"
        return body

    async def step(self) -> str | None:
        message, finish = self.chat()
        self.messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or ""
        for call in tool_calls:
            fn = call["function"]
            name, raw = fn["name"], fn.get("arguments") or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = None
            if arguments is None:
                envelope = {"schema_version": "wsp.tool/1.0", "ok": False,
                            "error": {"code": "invalid_json",
                                      "message": "tool arguments were not valid JSON"}}
            elif name not in self.tools:
                envelope = {"schema_version": "wsp.tool/1.0", "ok": False,
                            "error": {"code": "unknown_operation",
                                      "message": f"operation {name!r} does not exist in this server"}}
            else:
                self.log(f"  -> {name} {json.dumps(arguments)[:200]}")
                envelope = await self.call(name, arguments)
                if not envelope.get("ok", False):
                    code = (envelope.get("error") or {}).get("code", "?")
                    self.errors.append((name, code))
                    self.log(f"     ok=false code={code}")
                else:
                    self.log("     ok=true")
            self.messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": self.truncate(envelope),
            })
            if name == "discover_capabilities" and envelope.get("ok") and self.progressive:
                added = self.widen(envelope)
                if added:
                    self.log(f"  progressive disclosure: +{added} tools "
                             f"(total {len(self.tools)})")
        return None

    async def run(self) -> dict:
        started = time.time()
        await self.connect()
        self.messages = [
            {"role": "system", "content": (self.server_info["instructions"] or "") + "\n\n" + FRAMING},
            {"role": "user", "content": self.task},
        ]
        final = None
        try:
            for step in range(1, self.max_steps + 1):
                self.log(f"--- step {step} ---")
                final = await self.step()
                if final is not None:
                    break
        finally:
            await self.close()
        return {
            "task": self.task,
            "agent_model": self.model,
            "progressive": self.progressive,
            "server": self.server_info,
            "steps": len([m for m in self.messages if m.get("role") == "assistant"]),
            "tool_calls": [
                (tc["function"]["name"], (m.get("tool_calls") or [tc]).index(tc))
                for m in self.messages if m.get("role") == "assistant"
                for tc in (m.get("tool_calls") or [])
            ],
            "errors": self.errors,
            "usage": self.usage,
            "final_answer": final,
            "duration_s": round(time.time() - started, 1),
            "messages": self.messages,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--progressive", action="store_true")
    args = parser.parse_args()

    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        raise SystemExit("OPENCODE_API_KEY is not set")
    task = Path(args.task_file).read_text().strip()

    async def go():
        run = AgentRun(key=key, model=args.model, workspace=Path(args.workspace),
                       task=task, max_steps=args.max_steps, progressive=args.progressive)
        result = await run.run()
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n=== steps={result['steps']} errors={result['errors']} "
              f"cost=${result['usage']['cost']:.4f} ===")
        print("final:", (result["final_answer"] or "NO FINAL ANSWER")[:1000])

    asyncio.run(go())


if __name__ == "__main__":
    main()
