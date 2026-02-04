"""Generate benchmark report from evaluation metrics.

This script runs policy evaluation on nightly runs and generates a static HTML
dashboard for tracking policy performance over time.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import tyro
import wandb

import mjlab
from mjlab.tasks.tracking.scripts.evaluate import EvaluateConfig, run_evaluate

# Metrics to display: (key, label, unit, scale, higher_is_better)
METRICS = [
  ("success_rate", "Success Rate", "%", 100, True),
  ("mpkpe", "MPKPE", "m", 1, False),
  ("r_mpkpe", "R-MPKPE", "m", 1, False),
  ("ee_pos_error", "EE Position Error", "m", 1, False),
  ("ee_ori_error", "EE Orientation Error", "rad", 1, False),
  ("joint_vel_error", "Joint Velocity Error", "rad/s", 1, False),
]


def evaluate_run(run_path: str, num_envs: int = 1024) -> dict:
  """Evaluate a single run and return metrics with metadata."""
  api = wandb.Api()
  run = api.run(run_path)

  print(f"Evaluating run: {run.name} ({run.id})")

  cfg = EvaluateConfig(wandb_run_path=run_path, num_envs=num_envs)
  metrics = run_evaluate("Mjlab-Tracking-Flat-Unitree-G1", cfg)

  # Get commit SHA from run metadata.
  commit = run.commit or run.config.get("commit", "unknown")

  return {
    "id": run.id,
    "name": run.name,
    "url": run.url,
    "created_at": run.created_at,
    "commit": commit[:7] if len(commit) > 7 else commit,
    "metrics": metrics,
  }


def load_throughput_data(output_dir: Path) -> list[dict]:
  """Load throughput benchmark data if available."""
  data_file = output_dir / "throughput_data.json"
  if not data_file.exists():
    return []
  with open(data_file) as f:
    return json.load(f)


def generate_html_report(runs: list[dict], output_dir: Path) -> None:
  """Generate static HTML dashboard from evaluation data."""
  output_dir.mkdir(parents=True, exist_ok=True)

  # Save raw data.
  with open(output_dir / "data.json", "w") as f:
    json.dump(runs, f, indent=2, default=str)

  # Copy task images for the throughput dashboard.
  images_src = Path(__file__).parent / "nightly_images"
  if images_src.is_dir():
    images_dst = output_dir / "images"
    if images_dst.exists():
      shutil.rmtree(images_dst)
    shutil.copytree(images_src, images_dst)

  # Load throughput data if available.
  throughput_data = load_throughput_data(output_dir)

  html = generate_dashboard_html(runs, throughput_data)
  with open(output_dir / "index.html", "w") as f:
    f.write(html)

  print(f"Report generated at {output_dir / 'index.html'}")


def generate_dashboard_html(runs: list[dict], throughput_data: list[dict]) -> str:
  """Generate the HTML dashboard content."""
  runs_json = json.dumps(runs, default=str)
  metrics_json = json.dumps(METRICS)
  throughput_json = json.dumps(throughput_data, default=str)
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
  github_repo = "https://github.com/mujocolab/mjlab"

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MJLab Nightly Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        :root {{
            --bg: #ffffff;
            --bg-card: #f6f8fa;
            --text: #1f2328;
            --text-dim: #656d76;
            --border: #d0d7de;
            --accent: #0969da;
            --green: #1a7f37;
            --red: #cf222e;
        }}
        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                --bg: #0d1117;
                --bg-card: #161b22;
                --text: #c9d1d9;
                --text-dim: #8b949e;
                --border: #30363d;
                --accent: #58a6ff;
                --green: #3fb950;
                --red: #f85149;
            }}
        }}
        :root[data-theme="dark"] {{
            --bg: #0d1117;
            --bg-card: #161b22;
            --text: #c9d1d9;
            --text-dim: #8b949e;
            --border: #30363d;
            --accent: #58a6ff;
            --green: #3fb950;
            --red: #f85149;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 1.5rem; }}
        .subtitle {{
            font-size: 0.875rem;
            color: var(--text-dim);
            margin-top: 0.25rem;
        }}
        .timestamp {{ color: var(--text-dim); font-size: 0.875rem; }}
        .charts {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .chart-title {{
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}
        .chart-value {{ color: var(--text-dim); }}
        .chart-container {{ height: 350px; }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .theme-toggle {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem;
            cursor: pointer;
            color: var(--text);
            font-size: 1rem;
            line-height: 1;
        }}
        .theme-toggle:hover {{ border-color: var(--accent); }}
        .header-right {{ display: flex; align-items: center; gap: 1rem; }}
        .tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        .tab {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 1rem;
            cursor: pointer;
            color: var(--text);
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .tab:hover {{ border-color: var(--accent); }}
        .tab.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .tab-description {{
            font-size: 0.875rem;
            color: var(--text-dim);
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }}
        .task-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .task-card {{
            background: var(--bg-card);
            border: 2px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem;
            cursor: pointer;
            transition: border-color 0.15s, box-shadow 0.15s;
            text-align: center;
        }}
        .task-card:hover {{ border-color: var(--accent); }}
        .task-card.active {{
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }}
        .task-card img {{
            width: 100%;
            aspect-ratio: 16/10;
            object-fit: cover;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            background: var(--border);
        }}
        .task-card .task-name {{
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .task-card .task-stat {{
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 0.2rem;
        }}
        .task-chart-area {{
            display: none;
        }}
        .task-chart-area.active {{
            display: block;
        }}
        footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-dim);
            line-height: 1.6;
        }}
        @media (max-width: 600px) {{
            body {{ padding: 1rem; }}
            h1 {{ font-size: 1.25rem; }}
            .tabs {{ flex-wrap: wrap; }}
            .task-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1><a href="{github_repo}" style="color:inherit;text-decoration:none">MJLab</a> Nightly Benchmark</h1>
            <div class="subtitle">Performance tracking over time</div>
        </div>
        <div class="header-right">
            <span class="timestamp">Updated: {timestamp}</span>
            <button class="theme-toggle" id="theme-toggle" title="Toggle theme">
                <span id="theme-icon"></span>
            </button>
        </div>
    </header>

    <div class="tabs">
        <button class="tab active" data-tab="tracking">Tracking Eval</button>
        <button class="tab" data-tab="throughput">Throughput</button>
    </div>

    <div id="tracking" class="tab-content active">
        <p class="tab-description">Nightly motion imitation training and evaluation on Unitree G1 (1024 trials per run).</p>
        <div class="charts" id="charts"></div>
    </div>

    <div id="throughput" class="tab-content">
        <p class="tab-description">Physics simulation throughput across tasks (4096 parallel envs, NVIDIA RTX 5090).</p>
        <div class="task-grid" id="task-grid"></div>
        <div id="task-chart-panels"></div>
    </div>

    <footer>
        These benchmarks run nightly using the latest commit.<br>
        GPU: NVIDIA RTX 5090
    </footer>

    <script>
        // Theme toggle logic
        const themeToggle = document.getElementById('theme-toggle');
        const themeIcon = document.getElementById('theme-icon');
        const root = document.documentElement;

        function getSystemTheme() {{
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }}

        function getEffectiveTheme() {{
            const stored = localStorage.getItem('theme');
            if (stored === 'dark' || stored === 'light') return stored;
            return getSystemTheme();
        }}

        function updateThemeIcon() {{
            const stored = localStorage.getItem('theme');
            if (!stored) {{
                themeIcon.textContent = '\u2699\ufe0f'; // gear for auto
                themeToggle.title = 'Theme: System (click to toggle)';
            }} else if (stored === 'dark') {{
                themeIcon.textContent = '\U0001f319'; // moon
                themeToggle.title = 'Theme: Dark (click to toggle)';
            }} else {{
                themeIcon.textContent = '\u2600\ufe0f'; // sun
                themeToggle.title = 'Theme: Light (click to toggle)';
            }}
        }}

        function applyTheme() {{
            const stored = localStorage.getItem('theme');
            if (stored) {{
                root.setAttribute('data-theme', stored);
            }} else {{
                root.removeAttribute('data-theme');
            }}
            updateThemeIcon();
            updateChartColors();
        }}

        function cycleTheme() {{
            const stored = localStorage.getItem('theme');
            if (!stored) {{
                // auto -> dark
                localStorage.setItem('theme', 'dark');
            }} else if (stored === 'dark') {{
                // dark -> light
                localStorage.setItem('theme', 'light');
            }} else {{
                // light -> auto
                localStorage.removeItem('theme');
            }}
            applyTheme();
        }}

        themeToggle.addEventListener('click', cycleTheme);
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

        const runs = {runs_json};
        const METRICS = {metrics_json};
        const GITHUB_REPO = '{github_repo}';

        // Sort by date ascending for charts.
        runs.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

        const colors = {{
            success_rate: '#3fb950',
            mpkpe: '#58a6ff',
            r_mpkpe: '#a371f7',
            ee_pos_error: '#f0883e',
            ee_ori_error: '#f85149',
            joint_vel_error: '#79c0ff'
        }};

        let charts = [];

        function updateChartColors() {{
            const style = getComputedStyle(root);
            const textDim = style.getPropertyValue('--text-dim').trim();
            const border = style.getPropertyValue('--border').trim();
            const isDark = getEffectiveTheme() === 'dark';
            const gridColor = isDark ? '#3a424b' : '#d0d7de';
            Chart.defaults.color = textDim;
            Chart.defaults.borderColor = gridColor;
            charts.forEach(c => c.update());
        }}

        // Initialize theme before creating charts so grid colors are correct.
        applyTheme();

        // Charts
        const chartsContainer = document.getElementById('charts');

        // Compute a rolling average over the last `window` points.
        function rollingAvg(data, window) {{
            return data.map((d, i) => {{
                const start = Math.max(0, i - window + 1);
                const slice = data.slice(start, i + 1);
                const avg = slice.reduce((s, p) => s + p.y, 0) / slice.length;
                return {{ x: d.x, y: avg }};
            }});
        }}

        const AVG_WINDOW = 7;

        METRICS.forEach(([key, label, unit, scale, higherIsBetter]) => {{
            const data = runs.map(r => ({{
                x: new Date(r.created_at),
                y: r.metrics[key] * scale,
                commit: r.commit,
                name: r.name
            }}));

            const avgData = rollingAvg(data, AVG_WINDOW);
            const color = colors[key] || '#58a6ff';

            const latestVal = data[data.length - 1]?.y;
            const arrow = higherIsBetter ? '\u2191' : '\u2193';
            const tooltip = higherIsBetter ? 'Higher is better' : 'Lower is better';

            const card = document.createElement('div');
            card.className = 'chart-card';
            card.innerHTML = `
                <div class="chart-title">
                    <span>${{label}} <span title="${{tooltip}}" style="cursor:help;opacity:0.6">${{arrow}}</span></span>
                    <span class="chart-value">${{latestVal?.toFixed(3)}} ${{unit}}</span>
                </div>
                <div class="chart-container"><canvas></canvas></div>
            `;
            chartsContainer.appendChild(card);

            charts.push(new Chart(card.querySelector('canvas'), {{
                type: 'line',
                data: {{
                    datasets: [
                        {{
                            label: label,
                            data: data,
                            borderColor: color,
                            backgroundColor: color + '20',
                            borderWidth: 2,
                            pointRadius: 4,
                            tension: 0.1,
                            fill: true
                        }},
                        {{
                            label: `${{AVG_WINDOW}}-run avg`,
                            data: avgData,
                            borderColor: color,
                            borderWidth: 2,
                            borderDash: [6, 4],
                            pointRadius: 0,
                            tension: 0.3,
                            fill: false
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    onClick: (event, elements) => {{
                        if (elements.length > 0 && elements[0].datasetIndex === 0) {{
                            const d = data[elements[0].index];
                            if (d?.commit && d.commit !== 'unknown') {{
                                window.open(`${{GITHUB_REPO}}/commit/${{d.commit}}`, '_blank');
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'bottom',
                            labels: {{ usePointStyle: true, pointStyle: 'line', boxHeight: 1 }}
                        }},
                        tooltip: {{
                            filter: (item) => item.datasetIndex === 0,
                            callbacks: {{
                                title: (items) => {{
                                    const d = items[0]?.raw;
                                    return d ? `${{d.name}} (${{d.commit}})` : '';
                                }},
                                label: (item) => {{
                                    const d = item.raw;
                                    return `${{label}}: ${{d.y?.toFixed(4)}} ${{unit}}`;
                                }},
                                footer: (items) => {{
                                    const d = items[0]?.raw;
                                    return d?.commit && d.commit !== 'unknown' ? 'Click to view commit' : '';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{ unit: 'day' }},
                            ticks: {{ maxTicksLimit: 5 }},
                            title: {{
                                display: true,
                                text: 'Date',
                                font: {{ size: 11 }}
                            }}
                        }},
                        y: {{
                            ticks: {{ maxTicksLimit: 5 }},
                            title: {{
                                display: true,
                                text: unit,
                                font: {{ size: 11 }}
                            }}
                        }}
                    }}
                }}
            }}));
        }});

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
                // Update URL hash
                history.replaceState(null, '', '#' + tab.dataset.tab);
            }});
        }});

        // Handle URL hash on load
        if (window.location.hash) {{
            const tab = document.querySelector(`.tab[data-tab="${{window.location.hash.slice(1)}}"]`);
            if (tab) tab.click();
        }}

        // Throughput data and task cards
        const throughputData = {throughput_json};
        const taskGrid = document.getElementById('task-grid');
        const taskChartPanels = document.getElementById('task-chart-panels');

        // Task metadata: display name and image path (relative to nightly/)
        const taskMeta = {{
            'Mjlab-Velocity-Flat-Unitree-Go1': {{ name: 'Velocity \u2014 Go1', img: 'images/velocity_go1.png' }},
            'Mjlab-Tracking-Flat-Unitree-G1': {{ name: 'Tracking \u2014 G1', img: 'images/tracking_g1.png' }},
            'Mjlab-Lift-Cube-Yam': {{ name: 'Lift Cube \u2014 Yam', img: 'images/lift_cube_yam.png' }}
        }};

        const throughputChartInstances = {{}};

        if (throughputData.length > 0) {{
            const tasks = [...new Set(throughputData.flatMap(d => d.results.map(r => r.task)))];
            const latestRun = throughputData[throughputData.length - 1];

            tasks.forEach((task, i) => {{
                const meta = taskMeta[task] || {{ name: task, img: '' }};
                const latestResult = latestRun?.results.find(r => r.task === task);
                const latestSps = latestResult ? `${{(latestResult.env_sps / 1000).toFixed(0)}}K env steps/s` : '';

                // Card
                const card = document.createElement('div');
                card.className = 'task-card' + (i === 0 ? ' active' : '');
                card.dataset.task = task;
                card.innerHTML = `
                    <img src="${{meta.img}}" alt="${{meta.name}}" onerror="this.style.display='none'">
                    <div class="task-name">${{meta.name}}</div>
                    <div class="task-stat">${{latestSps}}</div>
                `;
                taskGrid.appendChild(card);

                // Chart panel
                const panel = document.createElement('div');
                panel.className = 'task-chart-area' + (i === 0 ? ' active' : '');
                panel.id = `task-panel-${{i}}`;
                panel.innerHTML = `
                    <div class="chart-card">
                        <div class="chart-title">
                            <span>${{meta.name}} \u2014 Throughput</span>
                        </div>
                        <div class="chart-container"><canvas></canvas></div>
                    </div>
                `;
                taskChartPanels.appendChild(panel);

                // Build datasets
                const envData = [];
                const physicsData = [];
                throughputData.forEach(run => {{
                    const result = run.results.find(r => r.task === task);
                    if (!result) return;
                    const point = {{ x: new Date(run.created_at), commit: run.commit }};
                    envData.push({{ ...point, y: result.env_sps / 1000 }});
                    physicsData.push({{ ...point, y: result.physics_sps / 1000 }});
                }});

                const chart = new Chart(panel.querySelector('canvas'), {{
                    type: 'line',
                    data: {{
                        datasets: [
                            {{
                                label: 'Env SPS',
                                data: envData,
                                borderColor: '#58a6ff',
                                backgroundColor: '#58a6ff20',
                                borderWidth: 2,
                                pointRadius: 4,
                                tension: 0.1,
                                fill: true
                            }},
                            {{
                                label: 'Physics SPS',
                                data: physicsData,
                                borderColor: '#3fb950',
                                backgroundColor: '#3fb95020',
                                borderWidth: 2,
                                pointRadius: 4,
                                tension: 0.1,
                                fill: true
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        onClick: (event, elements) => {{
                            if (elements.length > 0) {{
                                const di = elements[0].datasetIndex;
                                const idx = elements[0].index;
                                const d = di === 0 ? envData[idx] : physicsData[idx];
                                if (d?.commit && d.commit !== 'unknown') {{
                                    window.open(`${{GITHUB_REPO}}/commit/${{d.commit}}`, '_blank');
                                }}
                            }}
                        }},
                        plugins: {{
                            legend: {{ display: true, position: 'bottom' }},
                            tooltip: {{
                                mode: 'index',
                                callbacks: {{
                                    title: (items) => {{
                                        const d = items[0]?.raw;
                                        return d ? `Commit: ${{d.commit}}` : '';
                                    }},
                                    label: (item) => {{
                                        return `${{item.dataset.label}}: ${{item.raw.y?.toFixed(0)}}K steps/s`;
                                    }},
                                    afterBody: (items) => {{
                                        if (items.length >= 2) {{
                                            const envY = items[0]?.raw?.y || 0;
                                            const physY = items[1]?.raw?.y || 0;
                                            const overhead = physY > 0 ? ((1 - envY / physY) * 100).toFixed(1) : '?';
                                            return `Overhead: ${{overhead}}%`;
                                        }}
                                        return '';
                                    }},
                                    footer: (items) => {{
                                        const d = items[0]?.raw;
                                        return d?.commit && d.commit !== 'unknown' ? 'Click to view commit' : '';
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                type: 'time',
                                time: {{ unit: 'day' }},
                                ticks: {{ maxTicksLimit: 5 }},
                                title: {{ display: true, text: 'Date', font: {{ size: 11 }} }}
                            }},
                            y: {{
                                ticks: {{ maxTicksLimit: 5 }},
                                title: {{
                                    display: true,
                                    text: 'K steps/s',
                                    font: {{ size: 11 }}
                                }}
                            }}
                        }}
                    }}
                }});
                charts.push(chart);
                throughputChartInstances[task] = {{ chart, panelId: `task-panel-${{i}}` }};

                // Card click handler
                card.addEventListener('click', () => {{
                    document.querySelectorAll('.task-card').forEach(c => c.classList.remove('active'));
                    document.querySelectorAll('.task-chart-area').forEach(p => p.classList.remove('active'));
                    card.classList.add('active');
                    panel.classList.add('active');
                    // Trigger resize so chart renders at correct size
                    throughputChartInstances[task].chart.resize();
                }});
            }});
        }} else {{
            taskGrid.innerHTML = '<p style="color: var(--text-dim)">No throughput data available. Run measure_throughput.py to generate data.</p>';
        }}
    </script>
</body>
</html>
"""


def load_cached_results(output_dir: Path) -> dict[str, dict]:
  """Load previously evaluated results from cache."""
  data_file = output_dir / "data.json"
  if not data_file.exists():
    return {}

  with open(data_file) as f:
    runs = json.load(f)

  return {run["id"]: run for run in runs}


def main(
  run_paths: list[str] | None = None,
  entity: str = "gcbc_researchers",
  project: str = "mjlab",
  tag: str = "nightly",
  eval_limit: int = 0,
  num_envs: int = 1024,
  output_dir: Path = Path("benchmark_results"),
) -> None:
  """Generate benchmark report by evaluating nightly runs.

  Args:
    run_paths: Specific run paths to evaluate (entity/project/run_id).
    entity: WandB entity.
    project: WandB project name.
    tag: Filter runs by tag.
    eval_limit: Maximum number of NEW runs to evaluate per invocation (0 = no limit).
    num_envs: Number of envs for evaluation.
    output_dir: Output directory for generated report.
  """
  # Load cached results to avoid re-evaluating old runs.
  cached = load_cached_results(output_dir)
  print(f"Loaded {len(cached)} cached evaluation results")

  # Start with all cached results (preserves historical data).
  eval_results_by_id: dict[str, dict] = dict(cached)
  new_evals = 0

  if run_paths:
    for run_path in run_paths:
      run_id = run_path.split("/")[-1]
      if run_id in eval_results_by_id:
        print(f"Using cached result for {run_id}")
      else:
        result = evaluate_run(run_path, num_envs)
        eval_results_by_id[run_id] = result
        new_evals += 1
  else:
    api = wandb.Api()
    print(f"Fetching runs from {entity}/{project} with tag '{tag}'...")
    runs = api.runs(f"{entity}/{project}", filters={"tags": tag}, order="-created_at")

    for run in runs:
      if run.state != "finished":
        continue

      if run.id in eval_results_by_id:
        print(f"Using cached result for {run.name} ({run.id})")
      else:
        if eval_limit > 0 and new_evals >= eval_limit:
          print(f"Reached eval limit ({eval_limit}), skipping remaining new runs")
          break
        run_path = f"{entity}/{project}/{run.id}"
        result = evaluate_run(run_path, num_envs)
        eval_results_by_id[run.id] = result
        new_evals += 1

  eval_results = list(eval_results_by_id.values())
  print(f"Total runs: {len(eval_results)} ({new_evals} newly evaluated)")
  generate_html_report(eval_results, output_dir)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
