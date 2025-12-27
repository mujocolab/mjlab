"""Generate benchmark report from evaluation metrics.

This script runs policy evaluation on nightly runs and generates a static HTML
dashboard for tracking policy performance over time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import tyro
import wandb

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
    <title>MJLab Nightly Tracking Benchmark</title>
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
            max-width: 1200px;
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
        .timestamp {{ color: var(--text-dim); font-size: 0.875rem; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .stat-label {{ font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }}
        .stat-value.good {{ color: var(--green); }}
        .stat-value.bad {{ color: var(--red); }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
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
        .chart-container {{ height: 180px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
        }}
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
        @media (max-width: 600px) {{
            body {{ padding: 1rem; }}
            h1 {{ font-size: 1.25rem; }}
            .charts {{ grid-template-columns: 1fr; }}
            .summary {{ grid-template-columns: repeat(2, 1fr); }}
            table {{ display: block; overflow-x: auto; }}
            .tabs {{ flex-wrap: wrap; }}
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            color: var(--text-dim);
        }}
        .runs-details {{
            margin-top: 1.5rem;
        }}
        .runs-details summary {{
            cursor: pointer;
            padding: 0.75rem 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-weight: 500;
            color: var(--text-dim);
        }}
        .runs-details summary:hover {{
            border-color: var(--accent);
        }}
        .runs-details[open] summary {{
            border-radius: 8px 8px 0 0;
            border-bottom: none;
        }}
        .runs-details table {{
            border-radius: 0 0 8px 8px;
        }}
        .info-box {{
            margin-bottom: 1.5rem;
        }}
        .info-box summary {{
            cursor: pointer;
            font-size: 0.8rem;
            color: var(--accent);
            font-weight: 500;
        }}
        .info-box summary:hover {{
            text-decoration: underline;
        }}
        .info-box p {{
            margin-top: 0.75rem;
            padding: 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.875rem;
            line-height: 1.5;
            color: var(--text-dim);
        }}
    </style>
</head>
<body>
    <header>
        <h1>MJLab Nightly Benchmark</h1>
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
        <details class="info-box">
            <summary>What is this?</summary>
            <p>
              Every night, we train a policy to imitate a reference motion on the Unitree G1 using the latest MJLab commit.
              The trained policy is then evaluated across 1024 trials, with the results logged here to catch regressions over time.
            </p>
        </details>
        <div class="summary" id="summary"></div>
        <div class="charts" id="charts"></div>
        <details class="runs-details">
            <summary>All Runs</summary>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Commit</th>
                        <th>Run</th>
                        <th>Success Rate</th>
                        <th>MPKPE (m)</th>
                        <th>EE Pos Error (m)</th>
                    </tr>
                </thead>
                <tbody id="table-body"></tbody>
            </table>
        </details>
    </div>

    <div id="throughput" class="tab-content">
        <details class="info-box">
            <summary>What is this?</summary>
            <p>
                Measures simulation throughput in <strong>env steps per second</strong> for different tasks.
                <strong>Physics SPS</strong> is the theoretical max if only physics ran (no managers).
                <strong>Env SPS</strong> is the actual throughput including observations, rewards, terminations, and resets.
                <strong>Overhead</strong> is the percentage of time spent on non-physics work.
                Benchmarks run on NVIDIA RTX 5090 with 4096 parallel environments.
            </p>
            <p>
                Note: These benchmarks use zero actions, which causes frequent terminations and resets.
                This represents a worst-case scenario for overhead since resets are expensive.
            </p>
        </details>
        <div class="summary" id="throughput-summary"></div>
        <div class="charts" id="throughput-charts"></div>
        <details class="runs-details">
            <summary>All Runs</summary>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Commit</th>
                        <th>Task</th>
                        <th>Physics SPS</th>
                        <th>Env SPS</th>
                        <th>Overhead</th>
                    </tr>
                </thead>
                <tbody id="throughput-table-body"></tbody>
            </table>
        </details>
    </div>

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
            Chart.defaults.color = textDim;
            Chart.defaults.borderColor = border;
            charts.forEach(c => c.update());
        }}

        // Summary cards
        const summary = document.getElementById('summary');
        const latest = runs[runs.length - 1];
        if (latest) {{
            summary.innerHTML = `
                <div class="stat">
                    <div class="stat-label">Total Runs</div>
                    <div class="stat-value">${{runs.length}}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-value ${{latest.metrics.success_rate >= 0.95 ? 'good' : latest.metrics.success_rate < 0.8 ? 'bad' : ''}}">
                        ${{(latest.metrics.success_rate * 100).toFixed(1)}}%
                    </div>
                </div>
                <div class="stat">
                    <div class="stat-label">MPKPE</div>
                    <div class="stat-value">${{(latest.metrics.mpkpe * 100).toFixed(1)}} cm</div>
                </div>
                <div class="stat">
                    <div class="stat-label">EE Position Error</div>
                    <div class="stat-value">${{(latest.metrics.ee_pos_error * 100).toFixed(1)}} cm</div>
                </div>
            `;
        }}

        // Charts
        const chartsContainer = document.getElementById('charts');

        METRICS.forEach(([key, label, unit, scale, higherIsBetter]) => {{
            const data = runs.map(r => ({{
                x: new Date(r.created_at),
                y: r.metrics[key] * scale,
                commit: r.commit,
                name: r.name
            }}));

            const latestVal = data[data.length - 1]?.y;
            const arrow = higherIsBetter ? '↑' : '↓';
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
                    datasets: [{{
                        data: data,
                        borderColor: colors[key] || '#58a6ff',
                        backgroundColor: (colors[key] || '#58a6ff') + '20',
                        borderWidth: 2,
                        pointRadius: 4,
                        tension: 0.1,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    onClick: (event, elements) => {{
                        if (elements.length > 0) {{
                            const d = data[elements[0].index];
                            if (d?.commit && d.commit !== 'unknown') {{
                                window.open(`${{GITHUB_REPO}}/commit/${{d.commit}}`, '_blank');
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
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
                                text: `${{unit}} (${{higherIsBetter ? 'higher is better' : 'lower is better'}})`,
                                font: {{ size: 11 }}
                            }}
                        }}
                    }}
                }}
            }}));
        }});

        // Initialize theme
        applyTheme();

        // Table
        const tbody = document.getElementById('table-body');
        [...runs].reverse().forEach(run => {{
            const commitLink = run.commit && run.commit !== 'unknown'
                ? `<a href="${{GITHUB_REPO}}/commit/${{run.commit}}" target="_blank"><code>${{run.commit}}</code></a>`
                : `<code>${{run.commit}}</code>`;
            tbody.innerHTML += `
                <tr>
                    <td>${{new Date(run.created_at).toLocaleDateString()}}</td>
                    <td>${{commitLink}}</td>
                    <td><a href="${{run.url}}" target="_blank">${{run.name}}</a></td>
                    <td>${{(run.metrics.success_rate * 100).toFixed(1)}}%</td>
                    <td>${{run.metrics.mpkpe.toFixed(4)}}</td>
                    <td>${{run.metrics.ee_pos_error.toFixed(4)}}</td>
                </tr>
            `;
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

        // Throughput data and charts
        const throughputData = {throughput_json};
        const throughputChartsContainer = document.getElementById('throughput-charts');
        const throughputSummary = document.getElementById('throughput-summary');
        const throughputTbody = document.getElementById('throughput-table-body');

        if (throughputData.length > 0) {{
            // Get unique tasks from the data
            const tasks = [...new Set(throughputData.flatMap(d => d.results.map(r => r.task)))];

            // Throughput summary (latest values)
            const latestRun = throughputData[throughputData.length - 1];
            if (latestRun) {{
                let summaryHtml = `<div class="stat"><div class="stat-label">Total Runs</div><div class="stat-value">${{throughputData.length}}</div></div>`;
                latestRun.results.forEach(r => {{
                    const taskShort = r.task.replace('Mjlab-', '').replace('-Unitree-', '-');
                    summaryHtml += `
                        <div class="stat">
                            <div class="stat-label">${{taskShort}} Env SPS</div>
                            <div class="stat-value">${{(r.env_sps / 1000).toFixed(0)}}K</div>
                        </div>`;
                }});
                throughputSummary.innerHTML = summaryHtml;
            }}

            // Create charts for each metric type
            const throughputMetrics = [
                ['physics_sps', 'Physics SPS', true],
                ['env_sps', 'Env SPS', true],
                ['overhead_pct', 'Overhead %', false]
            ];

            const taskColors = {{
                'Mjlab-Velocity-Flat-Unitree-Go1': '#3fb950',
                'Mjlab-Tracking-Flat-Unitree-G1': '#58a6ff',
                'Mjlab-Lift-Cube-Yam': '#f0883e'
            }};

            throughputMetrics.forEach(([key, label, higherIsBetter]) => {{
                const card = document.createElement('div');
                card.className = 'chart-card';
                const arrow = higherIsBetter ? '↑' : '↓';
                const tooltip = higherIsBetter ? 'Higher is better' : 'Lower is better';
                card.innerHTML = `
                    <div class="chart-title">
                        <span>${{label}} <span title="${{tooltip}}" style="cursor:help;opacity:0.6">${{arrow}}</span></span>
                    </div>
                    <div class="chart-container"><canvas></canvas></div>
                `;
                throughputChartsContainer.appendChild(card);

                const datasets = tasks.map(task => {{
                    const data = throughputData.map(run => {{
                        const result = run.results.find(r => r.task === task);
                        if (!result) return null;
                        const value = result[key];
                        return {{
                            x: new Date(run.created_at),
                            y: key.includes('_sps') ? value / 1000 : value,
                            commit: run.commit
                        }};
                    }}).filter(d => d !== null);

                    const taskShort = task.replace('Mjlab-', '').replace('-Unitree-', '-');
                    return {{
                        label: taskShort,
                        data: data,
                        borderColor: taskColors[task] || '#58a6ff',
                        backgroundColor: (taskColors[task] || '#58a6ff') + '20',
                        borderWidth: 2,
                        pointRadius: 4,
                        tension: 0.1
                    }};
                }});

                charts.push(new Chart(card.querySelector('canvas'), {{
                    type: 'line',
                    data: {{ datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        onClick: (event, elements) => {{
                            if (elements.length > 0) {{
                                const datasetIndex = elements[0].datasetIndex;
                                const index = elements[0].index;
                                const d = datasets[datasetIndex].data[index];
                                if (d?.commit && d.commit !== 'unknown') {{
                                    window.open(`${{GITHUB_REPO}}/commit/${{d.commit}}`, '_blank');
                                }}
                            }}
                        }},
                        plugins: {{
                            legend: {{ display: true, position: 'bottom' }},
                            tooltip: {{
                                callbacks: {{
                                    title: (items) => {{
                                        const d = items[0]?.raw;
                                        return d ? `Commit: ${{d.commit}}` : '';
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
                                    text: key.includes('_sps') ? 'K env steps/sec' : '%',
                                    font: {{ size: 11 }}
                                }}
                            }}
                        }}
                    }}
                }}));
            }});

            // Throughput table
            [...throughputData].reverse().forEach(run => {{
                run.results.forEach((r, i) => {{
                    const taskShort = r.task.replace('Mjlab-', '').replace('-Unitree-', '-');
                    const commitLink = run.commit && run.commit !== 'unknown'
                        ? `<a href="${{GITHUB_REPO}}/commit/${{run.commit}}" target="_blank"><code>${{run.commit}}</code></a>`
                        : `<code>${{run.commit}}</code>`;
                    throughputTbody.innerHTML += `
                        <tr>
                            <td>${{i === 0 ? new Date(run.created_at).toLocaleDateString() : ''}}</td>
                            <td>${{i === 0 ? commitLink : ''}}</td>
                            <td>${{taskShort}}</td>
                            <td>${{(r.physics_sps / 1000).toFixed(0)}}K</td>
                            <td>${{(r.env_sps / 1000).toFixed(0)}}K</td>
                            <td>${{r.overhead_pct.toFixed(1)}}%</td>
                        </tr>
                    `;
                }});
            }});
        }} else {{
            throughputSummary.innerHTML = '<p style="color: var(--text-dim)">No throughput data available. Run measure_throughput.py to generate data.</p>';
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
  eval_limit: int = 10,
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
  tyro.cli(main)
