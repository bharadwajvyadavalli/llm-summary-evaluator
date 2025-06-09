"""HTML Report Generation Module"""

import pandas as pd
from pathlib import Path


class HTMLReportGenerator:
    def generate_professional_report(self, df, analysis, output_path):
        """Generate comprehensive HTML report with visualizations"""

        # Calculate performance distribution
        performance_dist = {
            'Excellent (8-10)': analysis['excellent_answers'],
            'Good (6-8)': analysis['good_answers'],
            'Poor (<6)': analysis['poor_answers']
        }

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Query Performance Report</title>
            <meta charset="utf-8">
            <style>
                {self._get_css_styles()}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>LLM Query Performance Report</h1>
                    <p>Comprehensive Analysis of Question-Answering System</p>
                </div>
            </div>

            <div class="container">
                {self._generate_summary_cards(analysis)}
                {self._generate_insights_section(analysis, df)}
                {self._generate_performance_section(performance_dist)}
                {self._generate_quality_section(analysis, df)}
                {self._generate_metrics_section(analysis, df)}
                {self._generate_top_questions_section(df)}
                {self._generate_bottom_questions_section(df)}
                {self._generate_correlation_section(analysis)}
            </div>

            <div class="footer">
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | LLM Evaluation System</p>
            </div>

            {self._generate_javascript(performance_dist, analysis)}
        </body>
        </html>
        """

        with open(output_path / 'professional_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

    def generate_query_report(self, results, output_path):
        """Generate query-specific HTML report"""
        df = pd.DataFrame(results)

        html = f"""
        <html>
        <head>
            <title>Query Evaluation Report</title>
            <style>
                body {{ font-family: Arial; margin: 40px; }}
                .question {{ background: #e3f2fd; padding: 10px; margin: 10px 0; }}
                .answer {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .quality-High {{ color: green; }}
                .quality-Medium {{ color: orange; }}
                .quality-Low {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>📝 Query Evaluation Report</h1>
            <p><strong>Total Questions:</strong> {len(df)}</p>

            {"".join([f'''
            <div class="question">
                <h3>Q: {row['question']}</h3>
                <div class="answer">
                    <p><strong>Answer:</strong> {row['answer'][:500]}...</p>
                    <p><strong>Quality:</strong> <span class="quality-{row['quality']}">{row['quality']}</span></p>
                    <p><strong>Relevance:</strong> {row.get('relevance_score', 0):.2f} | 
                       <strong>Coherence:</strong> {row.get('coherence_score', 0):.2f}</p>
                </div>
            </div>
            ''' for _, row in df.iterrows()])}
        </body>
        </html>
        """

        with open(f"{output_path}.html", 'w', encoding='utf-8') as f:
            f.write(html)

    def generate_training_report(self, data, output_path):
        """Generate training report HTML"""
        df = pd.DataFrame(data)

        quality_dist = df['quality'].value_counts() if 'quality' in df else {}

        # Check if this is training data with intended quality
        intended_quality_section = ""
        if 'intended_quality' in df.columns:
            intended_dist = df['intended_quality'].value_counts()
            intended_quality_section = f"""
            <h2>Training Data Distribution</h2>
            <p><strong>Summaries per Quality Level:</strong></p>
            <ul>
                {"".join([f'<li><strong>{k}:</strong> {v} summaries</li>' for k, v in intended_dist.items()])}
            </ul>

            <h2>Model Performance</h2>
            <p>How well the model clusters align with intended quality:</p>
            <table>
                <tr><th>Predicted</th><th>High</th><th>Medium</th><th>Low</th></tr>
                {self._generate_confusion_matrix(df)}
            </table>
            """

        html = f"""
        <html>
        <head>
            <title>LLM Evaluation Report</title>
            <style>
                body {{ font-family: Arial; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ color: #2196F3; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>📊 LLM Evaluation Report</h1>

            <h2>Summary Statistics</h2>
            <p><strong>Total Documents:</strong> {len(df) if 'intended_quality' not in df else len(df) // 3}</p>
            <p><strong>Total Summaries:</strong> {len(df)}</p>
            <p><strong>Average Score:</strong> {df['overall_score'].mean():.2f}/10</p>

            {intended_quality_section}

            <h2>Quality Distribution (Model Predictions)</h2>
            <ul>
                {"".join([f'<li><strong>{k}:</strong> {v} summaries</li>' for k, v in quality_dist.items()])}
            </ul>

            <h2>Top Performing Summaries</h2>
            <table>
                <tr><th>Document</th><th>Quality</th><th>Score</th></tr>
                {self._generate_table_rows(df.nlargest(5, 'overall_score')[['document', 'quality', 'overall_score']])}
            </table>
        </body>
        </html>
        """

        with open(f"{output_path}.html", 'w', encoding='utf-8') as f:
            f.write(html)

    def generate_index_report(self, stats, chunks_df, output_path):
        """Generate HTML report for vector index"""
        # Group chunks by source
        source_stats = chunks_df.groupby('source').agg({
            'chunk_index': 'count',
            'text_length': ['mean', 'sum']
        }).round(2)

        html = f"""
        <html>
        <head>
            <title>Vector Index Report</title>
            <style>
                body {{ font-family: Arial; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ color: #2196F3; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🔍 Vector Index Report</h1>

            <h2>Index Overview</h2>
            <ul>
                <li><strong>Total Documents:</strong> {stats['total_documents']}</li>
                <li><strong>Total Chunks:</strong> {stats['total_chunks']}</li>
                <li><strong>Embedding Dimensions:</strong> {stats['embedding_dimensions']}</li>
                <li><strong>Average Chunks per Document:</strong> {stats['total_chunks'] / stats['total_documents']:.1f}</li>
            </ul>

            <h2>Document Statistics</h2>
            <table>
                <tr>
                    <th>Document</th>
                    <th>Number of Chunks</th>
                    <th>Avg Chunk Length</th>
                    <th>Total Characters</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{doc}</td>
                    <td>{int(source_stats.loc[doc, ('chunk_index', 'count')])}</td>
                    <td>{source_stats.loc[doc, ('text_length', 'mean')]:.0f}</td>
                    <td>{source_stats.loc[doc, ('text_length', 'sum')]:.0f}</td>
                </tr>
                ''' for doc in source_stats.index])}
            </table>

            <h2>Ready for Queries</h2>
            <p>The vector index is now ready to answer questions about the indexed documents.</p>
            <p>You can use query mode to ask questions about these documents.</p>
        </body>
        </html>
        """

        with open(f"{output_path}.html", 'w', encoding='utf-8') as f:
            f.write(html)

    def _get_css_styles(self):
        """Return CSS styles for the report"""
        return """
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #2c3e50;
                line-height: 1.6;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 0;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }
            .summary-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .card {
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            }
            .card h3 {
                margin: 0 0 10px 0;
                color: #667eea;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .card .value {
                font-size: 2.5em;
                font-weight: 600;
                margin: 0;
                color: #2c3e50;
            }
            .card .label {
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .section {
                background: white;
                padding: 30px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .section h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #e0e6ed;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e0e6ed;
            }
            th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #2c3e50;
            }
            tr:hover {
                background-color: #f8f9fa;
            }
            .quality-High {
                color: #27ae60;
                font-weight: 600;
            }
            .quality-Medium {
                color: #f39c12;
                font-weight: 600;
            }
            .quality-Low {
                color: #e74c3c;
                font-weight: 600;
            }
            .chart-container {
                margin: 20px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .bar {
                height: 30px;
                background: linear-gradient(to right, #667eea, #764ba2);
                margin: 5px 0;
                border-radius: 5px;
                position: relative;
                transition: all 0.3s;
            }
            .bar:hover {
                opacity: 0.8;
            }
            .bar-label {
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%);
                color: white;
                font-weight: 600;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .metric-item {
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
            }
            .metric-item .name {
                font-size: 0.9em;
                color: #7f8c8d;
                margin-bottom: 5px;
            }
            .metric-item .value {
                font-size: 1.8em;
                font-weight: 600;
                color: #2c3e50;
            }
            .insights {
                background: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 20px;
                margin: 20px 0;
                border-radius: 0 8px 8px 0;
            }
            .insights h3 {
                margin: 0 0 10px 0;
                color: #2980b9;
            }
            .insights ul {
                margin: 10px 0;
                padding-left: 20px;
            }
            .insights li {
                margin: 5px 0;
            }
            .footer {
                text-align: center;
                padding: 30px;
                color: #7f8c8d;
                font-size: 0.9em;
            }
        """

    def _generate_summary_cards(self, analysis):
        """Generate summary cards HTML"""
        return f"""
        <div class="summary-cards">
            <div class="card">
                <h3>Total Questions</h3>
                <p class="value">{analysis['total_questions']}</p>
                <p class="label">Evaluated Queries</p>
            </div>
            <div class="card">
                <h3>Average Score</h3>
                <p class="value">{analysis['avg_overall_score']:.2f}</p>
                <p class="label">Out of 10</p>
            </div>
            <div class="card">
                <h3>High Quality</h3>
                <p class="value">{analysis['high_quality_pct']:.1f}%</p>
                <p class="label">Of All Responses</p>
            </div>
            <div class="card">
                <h3>System Performance</h3>
                <p class="value">{analysis['system_rating']}</p>
                <p class="label">Overall Rating</p>
            </div>
        </div>
        """

    def _generate_insights_section(self, analysis, df):
        """Generate insights section HTML"""
        return f"""
        <div class="insights">
            <h3>📊 Key Insights</h3>
            <ul>
                <li>The system achieved an average overall score of <strong>{analysis['avg_overall_score']:.2f}/10</strong></li>
                <li><strong>{analysis['excellent_answers']}</strong> responses ({analysis['excellent_answers'] / len(df) * 100:.1f}%) were rated as excellent (score ≥ 8)</li>
                <li>Relevance score averaged <strong>{analysis['avg_relevance']:.2f}/10</strong>, indicating {'strong' if analysis['avg_relevance'] >= 7 else 'moderate'} alignment with questions</li>
                <li>Semantic similarity averaged <strong>{analysis['avg_semantic_sim']:.3f}</strong>, showing {'high' if analysis['avg_semantic_sim'] >= 0.7 else 'moderate'} content matching</li>
                <li>Performance consistency is <strong>{analysis['performance_consistency']}</strong> (std dev: {analysis['std_overall_score']:.2f})</li>
            </ul>
        </div>
        """

    def _generate_performance_section(self, performance_dist):
        """Generate performance distribution section"""
        return """
        <div class="section">
            <h2>📈 Performance Distribution</h2>
            <div class="chart-container">
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
        </div>
        """

    def _generate_quality_section(self, analysis, df):
        """Generate quality breakdown section"""
        return f"""
        <div class="section">
            <h2>🎯 Quality Distribution</h2>
            <div class="chart-container">
                {"".join([f'''
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span class="quality-{quality}">{quality}</span>
                        <span>{count} ({count / len(df) * 100:.1f}%)</span>
                    </div>
                    <div style="background: #e0e6ed; border-radius: 5px; overflow: hidden;">
                        <div class="bar" style="width: {count / len(df) * 100}%;">
                            <span class="bar-label">{count}</span>
                        </div>
                    </div>
                </div>
                ''' for quality, count in analysis['quality_distribution'].items()])}
            </div>
        </div>
        """

    def _generate_metrics_section(self, analysis, df):
        """Generate detailed metrics section"""
        return f"""
        <div class="section">
            <h2>📊 Detailed Metrics Analysis</h2>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="name">Avg Relevance</div>
                    <div class="value">{analysis['avg_relevance']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="name">Avg Coherence</div>
                    <div class="value">{analysis['avg_coherence']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="name">Avg ROUGE-1</div>
                    <div class="value">{analysis['avg_rouge1']:.3f}</div>
                </div>
                <div class="metric-item">
                    <div class="name">Avg ROUGE-2</div>
                    <div class="value">{analysis['avg_rouge2']:.3f}</div>
                </div>
                <div class="metric-item">
                    <div class="name">Avg ROUGE-L</div>
                    <div class="value">{analysis['avg_rougel']:.3f}</div>
                </div>
                <div class="metric-item">
                    <div class="name">Avg Semantic Sim</div>
                    <div class="value">{analysis['avg_semantic_sim']:.3f}</div>
                </div>
            </div>
        </div>
        """

    def _generate_top_questions_section(self, df):
        """Generate top performing questions section"""
        return f"""
        <div class="section">
            <h2>🏆 Top Performing Questions</h2>
            <table>
                <tr>
                    <th>Question</th>
                    <th>Overall Score</th>
                    <th>Quality</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{df.iloc[idx]['question'][:100]}...</td>
                    <td>{df.iloc[idx]['overall_score']:.2f}</td>
                    <td class="quality-{df.iloc[idx]['quality']}">{df.iloc[idx]['quality']}</td>
                </tr>
                ''' for idx in df.nlargest(5, 'overall_score').index])}
            </table>
        </div>
        """

    def _generate_bottom_questions_section(self, df):
        """Generate questions needing improvement section"""
        return f"""
        <div class="section">
            <h2>⚠️ Questions Needing Improvement</h2>
            <table>
                <tr>
                    <th>Question</th>
                    <th>Overall Score</th>
                    <th>Issue</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{df.iloc[idx]['question'][:100]}...</td>
                    <td>{df.iloc[idx]['overall_score']:.2f}</td>
                    <td>{'Low Relevance' if df.iloc[idx]['relevance_score'] < 5 else 'Low Coherence' if df.iloc[idx]['coherence_score'] < 5 else 'Poor Semantic Match'}</td>
                </tr>
                ''' for idx in df.nsmallest(5, 'overall_score').index])}
            </table>
        </div>
        """

    def _generate_correlation_section(self, analysis):
        """Generate correlation analysis section"""
        return """
        <div class="section">
            <h2>🔗 Metric Correlations with Overall Score</h2>
            <div class="chart-container">
                <canvas id="correlationChart" width="400" height="300"></canvas>
            </div>
        </div>
        """

    def _generate_javascript(self, performance_dist, analysis):
        """Generate JavaScript for charts"""
        return f"""
        <script>
            // Performance Distribution Chart
            const perfCtx = document.getElementById('performanceChart').getContext('2d');
            new Chart(perfCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {list(performance_dist.keys())},
                    datasets: [{{
                        data: {list(performance_dist.values())},
                        backgroundColor: ['#27ae60', '#f39c12', '#e74c3c'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                padding: 20,
                                font: {{
                                    size: 14
                                }}
                            }}
                        }}
                    }}
                }}
            }});

            // Correlation Chart
            const corrCtx = document.getElementById('correlationChart').getContext('2d');
            const correlations = {list(analysis['metric_correlations'].values())};
            const labels = {list(analysis['metric_correlations'].keys())};

            new Chart(corrCtx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Correlation with Overall Score',
                        data: correlations,
                        backgroundColor: correlations.map(v => v > 0.7 ? '#27ae60' : v > 0.4 ? '#f39c12' : '#e74c3c'),
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_confusion_matrix(self, df):
        """Generate confusion matrix HTML for training report"""
        if 'intended_quality' not in df.columns:
            return ""

        # Create confusion matrix
        matrix = {}
        for intended in ['high', 'medium', 'low']:
            matrix[intended] = {}
            for predicted in ['High', 'Medium', 'Low']:
                count = len(df[(df['intended_quality'] == intended) & (df['quality'] == predicted)])
                matrix[intended][predicted] = count

        rows = []
        for predicted in ['High', 'Medium', 'Low']:
            row = f"<tr><td><strong>{predicted}</strong></td>"
            for intended in ['high', 'medium', 'low']:
                count = matrix[intended][predicted]
                total = df[df['intended_quality'] == intended].shape[0]
                pct = (count / total * 100) if total > 0 else 0
                row += f"<td>{count} ({pct:.0f}%)</td>"
            row += "</tr>"
            rows.append(row)

        return "".join(rows)

    def _generate_table_rows(self, df):
        """Helper to generate HTML table rows"""
        return "".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]:.2f}</td></tr>"
                        for row in df.values])