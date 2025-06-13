"""HTML Report Generation Module"""

import pandas as pd
from pathlib import Path


class HTMLReportGenerator:
    def generate_professional_report(self, df, analysis, output_path):
        """Generate comprehensive HTML report with modern dark theme"""

        # Calculate performance distribution
        performance_dist = {
            'Excellent': analysis['excellent_answers'],
            'Good': analysis['good_answers'],
            'Poor': analysis['poor_answers']
        }

        # Prepare data for charts - take last 30 questions for trend
        num_questions = min(30, len(df))
        recent_df = df.tail(num_questions).copy()

        # Ensure we have data
        if len(recent_df) == 0:
            print("Warning: No data available for report")
            recent_df = df.copy()  # Use all data if tail returns empty

        questions_list = recent_df['question'].tolist()
        scores_list = recent_df['overall_score'].tolist()

        # Debug print
        print(f"Debug: Creating chart with {len(scores_list)} data points")
        if len(scores_list) > 0:
            print(f"Debug: Score range: {min(scores_list):.2f} - {max(scores_list):.2f}")
            print(f"Debug: First few scores: {scores_list[:5]}")

        # Normalize scores if they're out of expected range
        max_expected_score = 10.0
        if len(scores_list) > 0 and max(scores_list) > max_expected_score * 1.5:
            print(f"Warning: Scores appear to be out of range. Max score: {max(scores_list)}")
            # Try to normalize - assuming they might be percentages or need scaling
            max_score = max(scores_list)
            scores_list = [score * 10.0 / max_score for score in scores_list]
            print(f"Normalized scores to 0-10 range")

        # Format for JavaScript - create simple arrays
        questions_labels = [f'"Q{i+1}"' for i in range(len(questions_list))]
        scores_values = [f'{score:.2f}' for score in scores_list]

        # Create JavaScript array strings
        labels_js = '[' + ', '.join(questions_labels) + ']'
        data_js = '[' + ', '.join(scores_values) + ']'

        # Calculate appropriate Y-axis max
        y_max = 10 if len(scores_list) == 0 else min(10, max(int(max(scores_list)) + 2, 10))

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Performance Analytics Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
            <style>
                {self._get_modern_css_styles()}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <!-- Header -->
                <div class="header">
                    <h1>LLM Performance Analytics</h1>
                    <p class="subtitle">Generated on {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <!-- Executive Summary -->
                <div class="executive-summary">
                    <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">Executive Summary</h2>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <div class="summary-value">{analysis['total_questions']}</div>
                            <div class="summary-label">Total Questions</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-value">{analysis['avg_overall_score']:.1f}</div>
                            <div class="summary-label">Average Score</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-value">{analysis['high_quality_pct']:.0f}%</div>
                            <div class="summary-label">High Quality</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-value">{analysis['system_rating']}</div>
                            <div class="summary-label">System Rating</div>
                        </div>
                    </div>
                </div>
                
                <!-- Key Insights -->
                <div class="insights-container">
                    <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">📊 Key Insights</h2>
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-icon">🎯</div>
                            <div class="insight-text">
                                <strong>{analysis['excellent_answers']}</strong> responses ({analysis['excellent_answers']/len(df)*100:.0f}%) achieved excellent scores (≥8.0)
                            </div>
                        </div>
                        <div class="insight-card">
                            <div class="insight-icon">📈</div>
                            <div class="insight-text">
                                Average relevance score of <strong>{analysis['avg_relevance']:.1f}/10</strong> indicates {'strong' if analysis['avg_relevance'] >= 7 else 'moderate'} alignment
                            </div>
                        </div>
                        <div class="insight-card">
                            <div class="insight-icon">🔗</div>
                            <div class="insight-text">
                                Semantic similarity averaging <strong>{analysis['avg_semantic_sim']:.2f}</strong> shows {'high' if analysis['avg_semantic_sim'] >= 0.7 else 'moderate'} content matching
                            </div>
                        </div>
                        <div class="insight-card">
                            <div class="insight-icon">📊</div>
                            <div class="insight-text">
                                Performance consistency is <strong>{analysis['performance_consistency']}</strong> (σ = {analysis['std_overall_score']:.2f})
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Metrics Cards -->
                <div class="metrics-grid">
                    <!-- Alignment & Coverage Metrics -->
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon">🎯</div>
                            <div class="metric-title">Alignment & Coverage</div>
                        </div>
                        <div class="metric-content">
                            <div class="metric-row">
                                <span class="metric-label">Alignment Score</span>
                                <span class="metric-value">{analysis['avg_alignment']:.3f}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Coverage Score</span>
                                <span class="metric-value">{analysis['avg_coverage']:.3f}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Semantic Similarity</span>
                                <span class="metric-value">{analysis['avg_semantic_sim']:.3f}</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- LLM Judge Scores -->
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon">⚖️</div>
                            <div class="metric-title">LLM Judge Scores</div>
                        </div>
                        <div class="metric-content">
                            <div class="metric-row">
                                <span class="metric-label">Relevance</span>
                                <span class="metric-value">{analysis['avg_relevance']:.1f}/10</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Coherence</span>
                                <span class="metric-value">{analysis['avg_coherence']:.1f}/10</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Accuracy</span>
                                <span class="metric-value">{analysis['avg_accuracy']:.1f}/10</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Quality Distribution -->
                    <div class="metric-card">
                        <div class="metric-header">
                            <div class="metric-icon">📊</div>
                            <div class="metric-title">Quality Distribution</div>
                        </div>
                        <div class="metric-content">
                            {self._generate_quality_bars(analysis['quality_distribution'], len(df))}
                        </div>
                    </div>
                </div>
                
                <!-- Score Trend Chart -->
                <div class="chart-container">
                    <div class="chart-header">
                        <h3 class="chart-title">Overall Score Trend (Last {num_questions} Questions)</h3>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="scoreTrendChart"></canvas>
                    </div>
                </div>
                
                <!-- Performance Distribution Chart -->
                <div class="chart-container">
                    <div class="chart-header">
                        <h3 class="chart-title">Performance Distribution</h3>
                    </div>
                    <div class="chart-wrapper">
                        <canvas id="performanceChart"></canvas>
                    </div>
                </div>
                
                <!-- Metric Correlations -->
                <div class="chart-container">
                    <div class="chart-header">
                        <h3 class="chart-title">Metric Correlations with Overall Score</h3>
                    </div>
                    <div class="features-grid">
                        {self._generate_correlation_cards(analysis['metric_correlations'])}
                    </div>
                </div>
                
                <!-- Top & Bottom Performers -->
                <div class="performers-section">
                    <div class="performers-grid">
                        <!-- Top Performers -->
                        <div class="performers-card">
                            <h3 class="performers-title">🏆 Top Performing Questions</h3>
                            <div class="performers-list">
                                {self._generate_top_performers(df)}
                            </div>
                        </div>
                        
                        <!-- Bottom Performers -->
                        <div class="performers-card">
                            <h3 class="performers-title">⚠️ Questions Needing Improvement</h3>
                            <div class="performers-list">
                                {self._generate_bottom_performers(df)}
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Footer -->
                <div class="footer">
                    <p>Powered by LLM Evaluation System • {pd.Timestamp.now().year}</p>
                </div>
            </div>
            
            <script>
                // Chart.js configuration
                Chart.defaults.color = '#888';
                Chart.defaults.borderColor = '#2a2a3e';
                
                console.log('Initializing charts...');
                
                // Wait for DOM to be fully loaded
                window.addEventListener('DOMContentLoaded', function() {{
                    // Score Trend Chart
                    const scoreTrendCtx = document.getElementById('scoreTrendChart');
                    if (scoreTrendCtx) {{
                        const chartData = {data_js};
                        const chartLabels = {labels_js};
                        
                        console.log('Chart data:', chartData);
                        console.log('Chart labels:', chartLabels);
                        
                        // Calculate appropriate Y-axis max based on data
                        const maxDataValue = Math.max(...chartData);
                        const yAxisMax = Math.ceil(maxDataValue * 1.1); // 10% padding
                        
                        const scoreTrendChart = new Chart(scoreTrendCtx.getContext('2d'), {{
                            type: 'line',
                            data: {{
                                labels: chartLabels,
                                datasets: [{{
                                    label: 'Overall Score',
                                    data: chartData,
                                    borderColor: '#667eea',
                                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                    borderWidth: 3,
                                    pointRadius: 5,
                                    pointBackgroundColor: '#667eea',
                                    pointBorderColor: '#fff',
                                    pointBorderWidth: 2,
                                    tension: 0.4,
                                    fill: true
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    legend: {{ display: false }},
                                    tooltip: {{
                                        backgroundColor: '#1a1a2e',
                                        titleColor: '#e0e0e0',
                                        bodyColor: '#e0e0e0',
                                        borderColor: '#667eea',
                                        borderWidth: 1,
                                        cornerRadius: 8,
                                        displayColors: false,
                                        callbacks: {{
                                            label: function(context) {{
                                                return 'Score: ' + context.parsed.y.toFixed(2);
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    x: {{
                                        grid: {{ color: '#2a2a3e' }},
                                        ticks: {{ 
                                            color: '#888',
                                            font: {{ size: 11 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{ color: '#2a2a3e' }},
                                        beginAtZero: true,
                                        max: yAxisMax,
                                        ticks: {{
                                            color: '#888'
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }} else {{
                        console.error('Score trend canvas not found!');
                    }}
                    
                    // Performance Distribution Chart
                    const performanceCtx = document.getElementById('performanceChart');
                    if (performanceCtx) {{
                        const performanceChart = new Chart(performanceCtx.getContext('2d'), {{
                            type: 'doughnut',
                            data: {{
                                labels: {list(performance_dist.keys())},
                                datasets: [{{
                                    data: {list(performance_dist.values())},
                                    backgroundColor: [
                                        'rgba(16, 185, 129, 0.8)',  // Excellent - Green
                                        'rgba(245, 158, 11, 0.8)',  // Good - Yellow
                                        'rgba(239, 68, 68, 0.8)'    // Poor - Red
                                    ],
                                    borderWidth: 0,
                                    hoverOffset: 10
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
                                            font: {{ size: 14 }},
                                            color: '#e0e0e0'
                                        }}
                                    }},
                                    tooltip: {{
                                        backgroundColor: '#1a1a2e',
                                        titleColor: '#e0e0e0',
                                        bodyColor: '#e0e0e0',
                                        borderColor: '#667eea',
                                        borderWidth: 1,
                                        cornerRadius: 8
                                    }}
                                }}
                            }}
                        }});
                    }}
                }});
            </script>
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

    def _get_modern_css_styles(self):
        """Return modern dark theme CSS styles"""
        return """
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f23;
                color: #e0e0e0;
                line-height: 1.6;
            }
            
            .dashboard {
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            /* Header */
            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                border-radius: 20px;
                padding: 3rem;
                margin-bottom: 2rem;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                position: relative;
                overflow: hidden;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -10%;
                width: 300px;
                height: 300px;
                background: rgba(255,255,255,0.05);
                border-radius: 50%;
            }
            
            .header h1 {
                font-size: 2.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: linear-gradient(to right, #fff, #e0e0e0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .header .subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
                color: #e0e0e0;
            }
            
            /* Executive Summary */
            .executive-summary {
                background: #1a1a2e;
                border-radius: 16px;
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid #2a2a3e;
            }
            
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-top: 1.5rem;
            }
            
            .summary-item {
                text-align: center;
            }
            
            .summary-value {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .summary-label {
                color: #888;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Insights */
            .insights-container {
                background: #1a1a2e;
                border-radius: 16px;
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid #2a2a3e;
            }
            
            .insights-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin-top: 1rem;
            }
            
            .insight-card {
                background: #242438;
                border-radius: 12px;
                padding: 1.5rem;
                display: flex;
                align-items: center;
                gap: 1rem;
                transition: all 0.3s ease;
            }
            
            .insight-card:hover {
                background: #2a2a3e;
                transform: translateY(-2px);
            }
            
            .insight-icon {
                font-size: 2rem;
                flex-shrink: 0;
            }
            
            .insight-text {
                color: #e0e0e0;
                font-size: 0.95rem;
                line-height: 1.5;
            }
            
            .insight-text strong {
                color: #667eea;
                font-weight: 700;
            }
            
            /* Metrics Grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            .metric-card {
                background: #1a1a2e;
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid #2a2a3e;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
                border-color: #667eea;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: linear-gradient(to right, #667eea, #764ba2);
            }
            
            .metric-header {
                display: flex;
                align-items: center;
                margin-bottom: 1.5rem;
            }
            
            .metric-icon {
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 1rem;
                font-size: 1.5rem;
            }
            
            .metric-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: #e0e0e0;
            }
            
            .metric-content {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            
            .metric-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid #2a2a3e;
            }
            
            .metric-row:last-child {
                border-bottom: none;
            }
            
            .metric-label {
                color: #888;
                font-size: 0.9rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
                font-weight: 700;
                color: #667eea;
            }
            
            /* Quality Bars */
            .quality-bar-container {
                margin: 0.5rem 0;
            }
            
            .quality-bar-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
            }
            
            .quality-bar-track {
                background: #2a2a3e;
                height: 8px;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .quality-bar-fill {
                height: 100%;
                border-radius: 4px;
                transition: width 1s ease-out;
            }
            
            .quality-High .quality-bar-fill {
                background: linear-gradient(to right, #10b981, #34d399);
            }
            
            .quality-Medium .quality-bar-fill {
                background: linear-gradient(to right, #f59e0b, #fbbf24);
            }
            
            .quality-Low .quality-bar-fill {
                background: linear-gradient(to right, #ef4444, #f87171);
            }
            
            /* Charts */
            .chart-container {
                background: #1a1a2e;
                border-radius: 16px;
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid #2a2a3e;
            }
            
            .chart-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
            }
            
            .chart-title {
                font-size: 1.3rem;
                font-weight: 600;
                color: #e0e0e0;
            }
            
            .chart-wrapper {
                position: relative;
                height: 300px;
            }
            
            /* Feature/Correlation Cards */
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1.5rem;
            }
            
            .feature-card {
                background: #242438;
                border: 1px solid #3a3a4e;
                border-radius: 12px;
                padding: 1.2rem;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .feature-card:hover {
                background: #2a2a3e;
                border-color: #667eea;
                transform: scale(1.05);
            }
            
            .feature-name {
                font-weight: 600;
                color: #e0e0e0;
                margin-bottom: 0.5rem;
                text-transform: capitalize;
            }
            
            .feature-count {
                font-size: 1.8rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .feature-label {
                color: #888;
                font-size: 0.8rem;
            }
            
            /* Performers Section */
            .performers-section {
                margin-bottom: 2rem;
            }
            
            .performers-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 1.5rem;
            }
            
            .performers-card {
                background: #1a1a2e;
                border-radius: 16px;
                padding: 2rem;
                border: 1px solid #2a2a3e;
            }
            
            .performers-title {
                font-size: 1.3rem;
                font-weight: 600;
                color: #e0e0e0;
                margin-bottom: 1.5rem;
            }
            
            .performers-list {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            
            .performer-item {
                background: #242438;
                border-radius: 12px;
                padding: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: all 0.3s ease;
            }
            
            .performer-item:hover {
                background: #2a2a3e;
                transform: translateX(5px);
            }
            
            .performer-question {
                flex: 1;
                color: #e0e0e0;
                font-size: 0.95rem;
                margin-right: 1rem;
            }
            
            .performer-score {
                font-size: 1.2rem;
                font-weight: 700;
                color: #667eea;
                white-space: nowrap;
            }
            
            /* Status Indicators */
            .status {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
            }
            
            .status-excellent {
                background: rgba(16, 185, 129, 0.2);
                color: #10b981;
            }
            
            .status-good {
                background: rgba(59, 130, 246, 0.2);
                color: #3b82f6;
            }
            
            .status-poor {
                background: rgba(239, 68, 68, 0.2);
                color: #ef4444;
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 3rem 2rem;
                color: #666;
                font-size: 0.9rem;
            }
            
            /* Responsive */
            @media (max-width: 768px) {
                .dashboard { padding: 1rem; }
                .header { padding: 2rem; }
                .header h1 { font-size: 2rem; }
                .metrics-grid { grid-template-columns: 1fr; }
                .performers-grid { grid-template-columns: 1fr; }
            }
            
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .metric-card, .chart-container, .insight-card {
                animation: fadeIn 0.6s ease-out;
            }
            
            /* Print Styles */
            @media print {
                body { background: white; color: black; }
                .metric-card, .chart-container { 
                    background: white; 
                    border: 1px solid #ddd;
                    break-inside: avoid;
                }
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
                <li><strong>{analysis['excellent_answers']}</strong> responses ({analysis['excellent_answers']/len(df)*100:.1f}%) were rated as excellent (score ≥ 8)</li>
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
                        <span>{count} ({count/len(df)*100:.1f}%)</span>
                    </div>
                    <div style="background: #e0e6ed; border-radius: 5px; overflow: hidden;">
                        <div class="bar" style="width: {count/len(df)*100}%;">
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

    def _generate_quality_bars(self, quality_dist, total):
        """Generate quality distribution bars"""
        html = ""
        for quality, count in quality_dist.items():
            percentage = (count / total) * 100
            html += f"""
            <div class="quality-bar-container quality-{quality}">
                <div class="quality-bar-header">
                    <span>{quality}</span>
                    <span>{count} ({percentage:.0f}%)</span>
                </div>
                <div class="quality-bar-track">
                    <div class="quality-bar-fill" style="width: {percentage}%"></div>
                </div>
            </div>
            """
        return html

    def _generate_correlation_cards(self, correlations):
        """Generate correlation metric cards"""
        html = ""
        metric_names = {
            'alignment_score': 'Alignment',
            'coverage_score': 'Coverage',
            'semantic_similarity': 'Semantic Sim',
            'relevance_score': 'Relevance',
            'coherence_score': 'Coherence',
            'accuracy_score': 'Accuracy'
        }

        for metric, correlation in correlations.items():
            display_name = metric_names.get(metric, metric)
            html += f"""
            <div class="feature-card">
                <div class="feature-name">{display_name}</div>
                <div class="feature-count">{correlation:.3f}</div>
                <div class="feature-label">correlation</div>
            </div>
            """
        return html

    def _generate_top_performers(self, df):
        """Generate top performing questions list"""
        html = ""
        for idx in df.nlargest(5, 'overall_score').index:
            question = df.iloc[idx]['question'][:80] + '...' if len(df.iloc[idx]['question']) > 80 else df.iloc[idx]['question']
            score = df.iloc[idx]['overall_score']
            quality = df.iloc[idx]['quality']

            status_class = 'excellent' if score >= 8 else 'good'
            html += f"""
            <div class="performer-item">
                <div class="performer-question">{question}</div>
                <div class="performer-score">
                    <span class="status status-{status_class}">{score:.1f}/10</span>
                </div>
            </div>
            """
        return html

    def _generate_bottom_performers(self, df):
        """Generate bottom performing questions list"""
        html = ""
        for idx in df.nsmallest(5, 'overall_score').index:
            question = df.iloc[idx]['question'][:80] + '...' if len(df.iloc[idx]['question']) > 80 else df.iloc[idx]['question']
            score = df.iloc[idx]['overall_score']

            # Identify main issue
            if df.iloc[idx]['relevance_score'] < 5:
                issue = 'Low Relevance'
            elif df.iloc[idx]['coherence_score'] < 5:
                issue = 'Low Coherence'
            else:
                issue = 'Poor Semantic Match'

            html += f"""
            <div class="performer-item">
                <div class="performer-question">{question}</div>
                <div class="performer-score">
                    <span class="status status-poor">{score:.1f}/10 - {issue}</span>
                </div>
            </div>
            """
        return html

    def _generate_modern_javascript(self, questions_json, scores_json, performance_dist, analysis, questions_list):
        """Generate JavaScript for modern charts"""
        # Create a tooltip mapping for the full questions
        tooltip_mapping = {}
        for i, q in enumerate(questions_list):
            tooltip_mapping[f'Q{i+1}'] = q[:100] + '...' if len(q) > 100 else q

        return f"""
        <script>
            // Chart.js configuration
            Chart.defaults.color = '#888';
            Chart.defaults.borderColor = '#2a2a3e';
            
            // Tooltip mapping for full questions
            const questionTooltips = {str(tooltip_mapping).replace("'", '"')};
            
            // Score Trend Chart
            const scoreTrendCtx = document.getElementById('scoreTrendChart').getContext('2d');
            const scoreTrendChart = new Chart(scoreTrendCtx, {{
                type: 'line',
                data: {{
                    labels: {questions_json},
                    datasets: [{{
                        label: 'Overall Score',
                        data: {scores_json},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        pointRadius: 5,
                        pointBackgroundColor: '#667eea',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            backgroundColor: '#1a1a2e',
                            titleColor: '#e0e0e0',
                            bodyColor: '#e0e0e0',
                            borderColor: '#667eea',
                            borderWidth: 1,
                            cornerRadius: 8,
                            displayColors: false,
                            callbacks: {{
                                title: function(context) {{
                                    const label = context[0].label;
                                    return questionTooltips[label] || label;
                                }},
                                label: function(context) {{
                                    return 'Score: ' + context.parsed.y.toFixed(2) + '/10';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{ color: '#2a2a3e' }},
                            ticks: {{ 
                                color: '#888',
                                font: {{ size: 11 }}
                            }}
                        }},
                        y: {{
                            grid: {{ color: '#2a2a3e' }},
                            beginAtZero: true,
                            max: 10,
                            ticks: {{
                                color: '#888',
                                stepSize: 2
                            }}
                        }}
                    }}
                }}
            }});
            
            // Performance Distribution Chart
            const performanceCtx = document.getElementById('performanceChart').getContext('2d');
            const performanceChart = new Chart(performanceCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {list(performance_dist.keys())},
                    datasets: [{{
                        data: {list(performance_dist.values())},
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.8)',  // Excellent - Green
                            'rgba(245, 158, 11, 0.8)',  // Good - Yellow
                            'rgba(239, 68, 68, 0.8)'    // Poor - Red
                        ],
                        borderWidth: 0,
                        hoverOffset: 10
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
                                font: {{ size: 14 }},
                                color: '#e0e0e0',
                                generateLabels: function(chart) {{
                                    const data = chart.data;
                                    return data.labels.map((label, i) => {{
                                        const value = data.datasets[0].data[i];
                                        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                                        const percentage = ((value / total) * 100).toFixed(1);
                                        return {{
                                            text: label + ' (' + value + ' - ' + percentage + '%)',
                                            fillStyle: data.datasets[0].backgroundColor[i],
                                            index: i
                                        }};
                                    }});
                                }}
                            }}
                        }},
                        tooltip: {{
                            backgroundColor: '#1a1a2e',
                            titleColor: '#e0e0e0',
                            bodyColor: '#e0e0e0',
                            borderColor: '#667eea',
                            borderWidth: 1,
                            cornerRadius: 8,
                            callbacks: {{
                                label: function(context) {{
                                    const label = context.label || '';
                                    const value = context.parsed;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return label + ': ' + value + ' (' + percentage + '%)';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
            
            // Add animation to quality bars after page load
            window.addEventListener('load', function() {{
                const qualityBars = document.querySelectorAll('.quality-bar-fill');
                qualityBars.forEach((bar, index) => {{
                    setTimeout(() => {{
                        bar.style.transition = 'width 1s ease-out';
                    }}, index * 100);
                }});
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