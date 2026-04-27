import datetime
from pathlib import Path

def generate_html_report(results, output_path, scan_dir):
    """Generates a premium dark-mode HTML report from triage results."""
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_images = len(results)
    high_risk_count = sum(1 for r in results if r["triage_score"] > 0.8)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ForenSURE-Net Forensic Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0f172a;
                --panel: #1e293b;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #38bdf8;
                --danger: #ef4444;
                --warning: #f59e0b;
                --success: #10b981;
                --border: #334155;
            }}
            body {{
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 40px;
                line-height: 1.6;
            }}
            .header {{
                text-align: center;
                margin-bottom: 50px;
            }}
            .header h1 {{
                font-size: 2.5rem;
                font-weight: 800;
                color: var(--accent);
                margin-bottom: 10px;
                letter-spacing: -1px;
            }}
            .summary-cards {{
                display: flex;
                gap: 20px;
                margin-bottom: 40px;
                justify-content: center;
            }}
            .card {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
                width: 200px;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            .card h3 {{
                font-size: 0.9rem;
                color: var(--text-muted);
                margin: 0 0 10px 0;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .card .value {{
                font-size: 2rem;
                font-weight: 800;
                color: var(--text);
                margin: 0;
            }}
            .high-risk {{ color: var(--danger) !important; }}
            
            table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                background: var(--panel);
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--border);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            th, td {{
                padding: 16px 20px;
                text-align: left;
                border-bottom: 1px solid var(--border);
            }}
            th {{
                background: rgba(0,0,0,0.2);
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--text-muted);
                font-weight: 600;
            }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover td {{ background: rgba(255,255,255,0.02); }}
            
            .score-bar {{
                height: 6px;
                background: var(--border);
                border-radius: 3px;
                overflow: hidden;
                margin-top: 6px;
            }}
            .score-fill {{
                height: 100%;
                border-radius: 3px;
            }}
            .filename {{
                font-weight: 600;
                color: var(--accent);
                word-break: break-all;
            }}
            .badge {{
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }}
            .badge-danger {{ background: rgba(239, 68, 68, 0.1); color: var(--danger); }}
            .badge-warning {{ background: rgba(245, 158, 11, 0.1); color: var(--warning); }}
            .badge-safe {{ background: rgba(16, 185, 129, 0.1); color: var(--success); }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ForenSURE-Net Triage Report</h1>
            <p style="color: var(--text-muted);">Automated Steganalysis & Reliability Assessment</p>
        </div>

        <div class="summary-cards">
            <div class="card">
                <h3>Target Directory</h3>
                <div class="value" style="font-size: 1.2rem; word-break: break-all;">{scan_dir}</div>
            </div>
            <div class="card">
                <h3>Images Scanned</h3>
                <div class="value">{total_images}</div>
            </div>
            <div class="card">
                <h3>High Risk Found</h3>
                <div class="value high-risk">{high_risk_count}</div>
            </div>
            <div class="card">
                <h3>Timestamp</h3>
                <div class="value" style="font-size: 1rem; margin-top: 15px;">{timestamp}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Filename</th>
                    <th>Risk Level</th>
                    <th>Triage Score</th>
                    <th>P(Stego)</th>
                    <th>Reliability</th>
                    <th>Uncertainty</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for rank, item in enumerate(results, 1):
        score = item["triage_score"]
        
        if score > 0.8:
            risk = "High Risk"
            badge = "badge-danger"
            color = "var(--danger)"
        elif score > 0.5:
            risk = "Suspicious"
            badge = "badge-warning"
            color = "var(--warning)"
        else:
            risk = "Likely Clean"
            badge = "badge-safe"
            color = "var(--success)"
            
        html_content += f"""
                <tr>
                    <td style="color: var(--text-muted); font-weight: 600;">#{rank}</td>
                    <td>
                        <div class="filename">{item["filename"]}</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted);">{item["filepath"]}</div>
                    </td>
                    <td><span class="badge {badge}">{risk}</span></td>
                    <td>
                        <div style="font-weight: 800;">{score:.4f}</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score*100}%; background: {color};"></div>
                        </div>
                    </td>
                    <td>{item["stego_probability"]:.4f}</td>
                    <td>{item["reliability_score"]:.4f}</td>
                    <td>{item["uncertainty_score"]:.6f}</td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
        
        <div style="text-align: center; margin-top: 40px; color: var(--text-muted); font-size: 0.85rem;">
            Generated by ForenSURE-Net. Reliability-Calibrated Steganalysis Triage System.
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
