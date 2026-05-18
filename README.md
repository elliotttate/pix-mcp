# pix-mcp

An MCP (Model Context Protocol) server for **Microsoft PIX** on Windows. It
wraps `pixtool.exe` so an LLM agent can launch and attach to D3D12 processes,
take GPU captures, and run **fine-grained queries** over the resulting
`.wpix` files — including answering questions like *"what SRV is bound at
root parameter 0 of event 2372?"* without you having to dump the entire
event list to CSV and grep through C++ export by hand.

> Why a wrapper? `pixtool.exe` is a real scriptable CLI, but its queries are
> coarse: it dumps the whole event list as CSV, the whole resource as bytes,
> the whole frame as a C++ project. `pix-mcp` indexes those coarse outputs
> into a SQLite DB beside the capture and parses the C++ export, so the LLM
> can ask precise questions instantly without re-running pixtool.

## What it can do

Wraps every command exposed by `pixtool.exe`:

| pixtool command | MCP tool |
| --- | --- |
| `launch` (one-shot, capture-and-exit) | `pix_capture_launched` |
| `launch` (long-running, F11 in-game) | `pix_launch_background` |
| `attach <pid>` + `take-capture` + `save-capture` | `pix_capture_attached` |
| `programmatic-capture` | `pix_capture_programmatic` |
| `take-capture` / `save-capture` | (composed inside the workflows above) |
| `open-capture` | `pix_open_capture` (and implicitly chained by every capture-targeting tool) |
| `save-event-list` | `pix_save_event_list` |
| `save-resource` | `pix_save_resource` |
| `save-screenshot` | `pix_save_screenshot` |
| `list-counters` | `pix_list_counters` |
| `save-high-frequency-counters` | `pix_save_high_frequency_counters` |
| `export-to-cpp` | `pix_export_to_cpp` |
| `recapture-region` | `pix_recapture_region` |
| `upgrade-gpu-capture` | `pix_upgrade_gpu_capture` |
| any other / chained command | `pix_raw` |

And then the value-add — tools that **don't have a pixtool equivalent**
because they operate on an indexed view of the capture:

| Tool | What it answers |
| --- | --- |
| `pix_index_events` | Build a SQLite index from `save-event-list` CSV. |
| `pix_find_events` | "All draws inside marker `ShadowPass`", "all dispatches between events 2000 and 2400", "every `ResourceBarrier` on queue 1". |
| `pix_get_event` | Full per-event record including counters, by Global ID. |
| `pix_list_markers` | All PIX markers with their `[start, end]` Global IDs. |
| `pix_marker_range` | Resolve a marker name to its Global ID range. |
| `pix_list_queues` | Queues seen, with event counts. |
| `pix_event_type_counts` | Histogram (draws vs dispatches vs barriers vs copies …). |
| `pix_top_by_counter` | Top N events sorted by a numeric counter (e.g. `gpu_duration`). |
| `pix_query_sql` | Read-only SELECT/WITH against the index DB. |
| `pix_parse_cpp_export` | Parse an existing `export-to-cpp` directory. |
| `pix_state_at_event` | Replay the C++ export and return every bound state at a target event. |
| `pix_get_resource_at_root_param` | "what's at graphics root param N of event G?" — includes structured `resource_id` + `offset` when the binding is `GetGpuva(rid, off)`. |
| `pix_find_cpp_calls` | Search the parsed C++ export for specific D3D12 calls and their *exact* arguments. |
| `pix_get_root_signature_layout` | Parse the inline `D3D12_ROOT_PARAMETER1` array and return per-slot `{kind, visibility, register, register_space, descriptor_ranges, flags}` for a given root sig ApiObjectId. |
| `pix_get_resource_bytes` | Read raw bytes from a resource at `(offset, length)` by decompressing `resources.bin` (XPRESS via Windows Cabinet API). Solves the "save-resource only does PNG/DDS" gap. |
| `pix_list_tracked_resources` | List every resource ID `pix_get_resource_bytes` can read, with chunk sizes and source-function names. |
| `pix_dump_cbuffer_at_root_param` | End-to-end "show me these cbuffer bytes": composes state replay + GpuVa parsing + raw-bytes extraction. Returns hex + optional float[] decoding. |
| `pix_capture_summary` | One-shot high-level overview of a capture. |

## Install

```powershell
git clone <this repo> pix-mcp
cd pix-mcp
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Requires **Windows** and **Microsoft PIX** installed (default location
`C:\Program Files\Microsoft PIX\<version>\pixtool.exe`). pix-mcp auto-discovers
the newest installed version; override with `PIX_MCP_PIXTOOL=<full path>`.

Quick smoke test:

```powershell
pix-mcp --help
.venv\Scripts\python.exe -c "from pix_mcp.config import find_pixtool; print(find_pixtool())"
```

## Wiring it into Claude Code / Claude Desktop

Add to your MCP config (`claude_desktop_config.json` or the CLI equivalent):

```json
{
  "mcpServers": {
    "pix": {
      "command": "E:/github/pix-mcp/.venv/Scripts/pix-mcp.exe",
      "args": [],
      "env": {
        "PIX_MCP_PIXTOOL": "C:/Program Files/Microsoft PIX/2603.25/pixtool.exe"
      }
    }
  }
}
```

(Set `PIX_MCP_PIXTOOL` only if auto-discovery isn't finding the right install.)

## Typical workflows

### 1. Capture a launched workload, then analyze it

```
pix_capture_launched(
    exe="C:/games/MyGame.exe",
    args="-windowed",
    output_wpix="C:/captures/mygame.wpix",
    frames=1,
)

pix_open_capture("C:/captures/mygame.wpix")
pix_index_events(capture="<handle>", counter_groups=["D3D:*"])
pix_capture_summary(capture="<handle>")
```

### 2. Interactive game session: launch, press F11 in-game, find the capture

```
pix_launch_background(exe="C:/games/MyGame.exe", capture_key="F11")
# ...play to the spot, press F11...
pix_find_recent_captures()
pix_open_capture("<path returned above>")
pix_index_events(capture="<handle>")
```

### 3. Fine-grained query: "what SRV is bound at root param 0 of event 2372?"

```
pix_open_capture("C:/captures/mygame.wpix")
pix_export_to_cpp(capture="<handle>", output_dir="C:/captures/mygame_cpp",
                  use_winpixeventruntime=True, use_agility_sdk=True)
pix_get_resource_at_root_param(capture="<handle>", global_id=2372, root_param_index=0)
# →
# {
#   "global_id": 2372,
#   "root_param_index": 0,
#   "pipeline": "graphics",
#   "binding": { "kind": "descriptor_table", "handle": "...", "raw": "..." },
#   "graphics_pso": "...",
#   "graphics_root_signature": "...",
#   "descriptor_heaps": ["..."]
# }
```

For the full bound state at an event (PSO, root sig, all root params, IB/VB,
RTVs, viewports, scissors), use `pix_state_at_event`.

### 4. Find every draw inside a PIX marker

```
pix_marker_range(capture="<h>", marker_name="ShadowPass")
# {"found": true, "start": 2100, "end": 2480}

pix_find_events(capture="<h>", inside_marker="ShadowPass", event_type="draw", limit=50)
```

### 5. Top GPU-time draws

```
pix_index_events(capture="<h>", counters=["GPU Duration"])
pix_top_by_counter(capture="<h>", counter="gpu_duration", n=20, event_type="draw")
```

### 6. Recapture just the suspicious region for faster iteration

```
pix_recapture_region(capture="<h>", output_wpix="C:/captures/mygame_slice.wpix", start=2300, end=2600)
```

### 7. Dump raw cbuffer bytes from a draw call (no PIX UI)

```
pix_export_to_cpp(capture="<h>", output_dir="C:/captures/mygame_cpp",
                  use_winpixeventruntime=True, use_agility_sdk=True)

# One call replaces ~10 PIX-UI clicks:
pix_dump_cbuffer_at_root_param(
    capture="<h>", global_id=2372, root_param_index=3,
    length=256, preview_floats=64,
    output_file="C:/scratch/basepass_cbuffer.bin",
)
# → {
#     "resource_id": 2362,
#     "offset": 2259712,
#     "hex_preview": "00 00 80 3b ...",
#     "preview_floats": [0.00390625, 0.0, ...],
#     "output_file": "C:/scratch/basepass_cbuffer.bin"
# }
```

For lower-level access:

```
pix_get_root_signature_layout(capture="<h>", root_sig_obj_id=2361)
# → per-slot {kind, visibility, register, ranges, flags}

pix_get_resource_bytes(capture="<h>", resource_id=2362,
                       offset=2259712, length=256)
pix_list_tracked_resources(capture="<h>")
```

### 8. Escape hatch — chain arbitrary pixtool commands

```
pix_raw(args=[
  "open-capture", "C:/captures/mygame.wpix",
  "save-event-list", "C:/scratch/events.csv", "--counters=*",
  "list-counters",
])
```

## How the index works

The first time you call `pix_index_events`, pix-mcp runs:

```
pixtool open-capture <wpix> save-event-list <cache>/<stem>.event_list.csv [--counters=...]
```

Then it streams the CSV into `<wpix>.pixmcp.db` (next to the .wpix, or in
`%LOCALAPPDATA%\pix-mcp\cache\` if that directory isn't writable). Schema:

```sql
events(global_id, queue_id, queue, name, event_type, parent_marker_id, depth,
       counters JSON, raw JSON)
markers(id, name, start_global_id, end_global_id, parent_id, depth, queue_id, queue)
queues(queue_id, name, event_count)
counter_columns(name)
```

`event_type` is a classified label (`draw`, `dispatch`, `copy`, `barrier`,
`begin_marker`, `end_marker`, `marker`, `clear`, `resolve`, `present`,
`execute_command_lists`, `execute_indirect`, `build_as`, `other`). Markers are
reconstructed from PIX's `BeginEvent`/`EndEvent` rows with per-queue depth
tracking — so `pix_find_events(inside_marker="ShadowPass")` works.

`counters` is a JSON object — query it via SQL with
`json_extract(counters, '$.gpu_duration')`. `raw` is the entire original CSV
row for escape-hatch queries.

The index is keyed by the wpix's mtime so it auto-invalidates when you re-capture.

## How the state-at-event query works

`pix_export_to_cpp(parse_after=True)` runs:

```
pixtool open-capture <wpix> export-to-cpp <dir> [--force --use-winpixeventruntime ...]
```

then immediately parses every `.cpp` under `<dir>`, picking out D3D12 calls
with regex. For each call we record `(call_index, global_id, method, args,
source_file, source_line)`. `global_id` is recovered from PIX's
`// Event <id>` comments where present.

`pix_state_at_event` walks that list up to your target event and replays the
state: graphics + compute root signatures, PSO, root parameters (descriptor
tables, root CBV/SRV/UAV, root 32-bit constants), descriptor heaps, IB/VB
views, RTVs/DSV, viewports/scissors, primitive topology.

For the specific "resource at root param N of event G" question,
`pix_get_resource_at_root_param` returns the *exact* `args` string PIX
emitted for the last `Set*Root*` call targeting that slot — so you get the
descriptor table base / GPU VA / 32-bit constant payload PIX serialized.

## Caveats

- **Windows only.** The bridge to `pixtool.exe` is Win32-native.
- **PIX must be installed.** Set `PIX_MCP_PIXTOOL` if auto-discovery fails.
- The C++-export parser is regex-based and tolerates PIX's formatting variants
  — it doesn't compile the export. If a future PIX version reshapes the
  export significantly, parsing of *unrecognized* methods will fall back to
  raw call records (still queryable via `pix_find_cpp_calls`).
- "Global ID" in the event list isn't guaranteed to match the call index in
  the C++ export 1-to-1. We rely on PIX's `// Event <id>` comments inside the
  export to associate them. If your PIX version doesn't emit those comments,
  pass `call_index=` instead of `global_id=` to `pix_state_at_event` and use
  `pix_find_cpp_calls` to discover the right call.
- `pix_capture_attached` requires WinPix services running; some games block
  attach until you've launched them through the PIX UI once.

## Layout

```
src/pix_mcp/
  config.py        # PIX install discovery, cache paths
  pixtool.py       # pixtool.exe subprocess wrapper (sync + async + background)
  csv_parser.py    # streaming parser for save-event-list CSV
  index.py         # SQLite indexer + query helpers
  cpp_export.py    # parser + state replay for export-to-cpp;
                   # also root-signature-layout extraction
  resources_bin.py # static call-graph walk over CreateAndInitResource_*
                   # functions + XPRESS decompression of resources.bin
                   # (uses Windows Cabinet API via ctypes)
  session.py       # in-memory capture / launch session registry
  server.py        # FastMCP server with all tools
  __main__.py      # console entry point
```

## License

MIT.
