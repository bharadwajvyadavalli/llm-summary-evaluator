"""Report Generation and Analysis Module"""

import pandas as pd
import numpy as np
from pathlib import Path
from html_generator import HTMLReportGenerator


class ReportGenerator:
    def __init__(self):
        self.html_generator = HTMLReportGenerator()

    def generate_reports(self, input_file, output_path):
        """Generate all reports from query results"""
        # Read query results
        df = pd.read_csv(input_file)
        print(f"📖 Loaded {len(df)} query results from {input_file}")

        # Generate comprehensive analysis
        analysis = self.analyze_query_results(df)

        # Generate CSV reports
        self.generate_csv_reports(df, analysis, output_path)

        # Generate HTML report
        self.html_generator.generate_professional_report(df, analysis, output_path)

        return analysis

    def analyze_query_results(self, df):
        """Perform comprehensive analysis of query results"""
        analysis = {}

        # Overall statistics
        analysis['total_questions'] = len(df)
        analysis['avg_overall_score'] = df['overall_score'].mean()
        analysis['avg_relevance'] = df['relevance_score'].mean()
        analysis['avg_coherence'] = df['coherence_score'].mean()
        analysis['avg_rouge1'] = df['rouge_1'].mean()
        analysis['avg_rouge2'] = df['rouge_2'].mean()
        analysis['avg_rougel'] = df['rouge_l'].mean()
        analysis['avg_semantic_sim'] = df['semantic_similarity'].mean()

        # Quality distribution
        quality_counts = df['quality'].value_counts()
        analysis['quality_distribution'] = quality_counts.to_dict()
        analysis['high_quality_pct'] = (quality_counts.get('High', 0) / len(df) * 100)

        # Performance categories
        analysis['excellent_answers'] = len(df[df['overall_score'] >= 8])
        analysis['good_answers'] = len(df[(df['overall_score'] >= 6) & (df['overall_score'] < 8)])
        analysis['poor_answers'] = len(df[df['overall_score'] < 6])

        # Top and bottom performers
        analysis['top_questions'] = df.nlargest(5, 'overall_score')[['question', 'overall_score', 'quality']].to_dict('records')
        analysis['bottom_questions'] = df.nsmallest(5, 'overall_score')[['question', 'overall_score', 'quality']].to_dict('records')

        # Metric correlations
        metrics_cols = ['rouge_1', 'rouge_2', 'rouge_l', 'semantic_similarity', 'relevance_score', 'coherence_score']
        if len(df) > 1:  # Need at least 2 rows for correlation
            # Include overall_score in the correlation calculation
            corr_matrix = df[metrics_cols + ['overall_score']].corr()
            analysis['metric_correlations'] = corr_matrix['overall_score'][metrics_cols].to_dict()
        else:
            analysis['metric_correlations'] = {col: 0 for col in metrics_cols}

        # Additional insights
        analysis['std_overall_score'] = df['overall_score'].std()
        analysis['performance_consistency'] = 'High' if analysis['std_overall_score'] < 1.5 else 'Medium' if analysis['std_overall_score'] < 2.5 else 'Low'

        # System rating
        avg_score = analysis['avg_overall_score']
        analysis['system_rating'] = 'Excellent' if avg_score >= 8 else 'Good' if avg_score >= 6 else 'Needs Improvement'

        return analysis

    def generate_csv_reports(self, df, analysis, output_path):
        """Generate multiple CSV reports for different analyses"""

        # 1. Metrics Summary Report
        metrics_summary = pd.DataFrame({
            'Metric': [
                'Overall Score',
                'Relevance Score',
                'Coherence Score',
                'ROUGE-1',
                'ROUGE-2',
                'ROUGE-L',
                'Semantic Similarity'
            ],
            'Average': [
                analysis['avg_overall_score'],
                analysis['avg_relevance'],
                analysis['avg_coherence'],
                analysis['avg_rouge1'],
                analysis['avg_rouge2'],
                analysis['avg_rougel'],
                analysis['avg_semantic_sim']
            ],
            'Std Dev': [
                df['overall_score'].std(),
                df['relevance_score'].std(),
                df['coherence_score'].std(),
                df['rouge_1'].std(),
                df['rouge_2'].std(),
                df['rouge_l'].std(),
                df['semantic_similarity'].std()
            ],
            'Min': [
                df['overall_score'].min(),
                df['relevance_score'].min(),
                df['coherence_score'].min(),
                df['rouge_1'].min(),
                df['rouge_2'].min(),
                df['rouge_l'].min(),
                df['semantic_similarity'].min()
            ],
            'Max': [
                df['overall_score'].max(),
                df['relevance_score'].max(),
                df['coherence_score'].max(),
                df['rouge_1'].max(),
                df['rouge_2'].max(),
                df['rouge_l'].max(),
                df['semantic_similarity'].max()
            ]
        })
        metrics_summary.round(3).to_csv(output_path / 'metrics_summary.csv', index=False)

        # 2. Quality Breakdown Report
        quality_data = []
        for level, count in analysis['quality_distribution'].items():
            quality_data.append({
                'Quality Level': level,
                'Count': count,
                'Percentage': f"{count/len(df)*100:.1f}%",
                'Avg Score': df[df['quality'] == level]['overall_score'].mean()
            })
        quality_breakdown = pd.DataFrame(quality_data)
        quality_breakdown.to_csv(output_path / 'quality_breakdown.csv', index=False)

        # 3. Question Performance Report
        question_perf = df[['question', 'quality', 'overall_score', 'relevance_score', 'coherence_score']].copy()
        question_perf['performance_category'] = pd.cut(
            df['overall_score'],
            bins=[0, 6, 8, 10],
            labels=['Needs Improvement', 'Good', 'Excellent']
        )
        question_perf['answer_preview'] = df['answer'].str[:100] + '...'
        question_perf = question_perf.sort_values('overall_score', ascending=False)
        question_perf.round(2).to_csv(output_path / 'question_performance.csv', index=False)

        # 4. Detailed Metrics Report (all metrics for each question)
        detailed_metrics = df.copy()
        # Remove chunks column as it's too large for CSV
        if 'chunks' in detailed_metrics.columns:
            detailed_metrics = detailed_metrics.drop('chunks', axis=1)
        detailed_metrics = detailed_metrics.sort_values('overall_score', ascending=False)
        detailed_metrics.round(3).to_csv(output_path / 'detailed_metrics.csv', index=False)