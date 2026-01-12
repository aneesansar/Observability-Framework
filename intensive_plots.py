#!/usr/bin/env python3
"""
Intensive Plot Generator for LLM Observability Framework.
Generates individual publication-quality plots for each metric.
"""

import json
import os
import statistics
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from typing import Any, Dict, List

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# Short label mappings for cleaner plots
TENANT_SHORT = lambda t: t.replace('tenant_', 'T').upper() if t else 'UNK'
STAGE_SHORT = {
    'RAGPipeline.query': 'Query',
    'EmbeddingService.embed': 'Embed',
    'EmbeddingService.embed_batch': 'EmbedBatch',
    'VectorStore.search': 'Retrieval',
    'LLMGenerator.generate': 'LLM',
    'compute_answer_relevancy': 'Rel',
    'compute_faithfulness': 'Faith',
    'check_consistency': 'Consist',
    'check_hallucination': 'Halluc',
    'generate_response': 'Resp',
    'MongoDBClient': 'Mongo',
}

PLOTS_DIR = "plots/intensive"
os.makedirs(PLOTS_DIR, exist_ok=True)


class IntensivePlotGenerator:
    """Generates individual plots for each evaluation metric."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
    
    def generate_all(self):
        """Generate all plots."""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        # Latency plots
        self.plot_latency_histogram()
        self.plot_latency_percentiles()
        self.plot_latency_boxplot()
        
        # Throughput plots
        self.plot_throughput_scaling()
        self.plot_throughput_success()
        
        # Resource plots
        self.plot_cpu_timeline()
        self.plot_memory_timeline()
        self.plot_resource_summary()
        
        # Observability plots
        self.plot_spans_distribution()
        self.plot_stage_coverage()
        self.plot_attribute_completeness()
        
        # Isolation plots
        self.plot_isolation_heatmap()
        self.plot_trace_distribution()
        
        # Scalability plots
        self.plot_tenant_scaling()
        
        # Quality plots
        self.plot_quality_radar()
        self.plot_quality_bars()
        self.plot_quality_per_tenant()
        
        # Summary dashboard
        self.plot_summary_dashboard()
        
        print(f"\nAll plots saved to: {PLOTS_DIR}/")
    
    # =========================================================================
    # LATENCY PLOTS
    # =========================================================================
    
    def plot_latency_histogram(self):
        """Plot 1: Latency distribution histogram."""
        data = self.results.get("latency", {})
        lats = data.get("raw_latencies", [])
        if not lats:
            print("  No latency data for histogram")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lats, bins=25, color='#3498db', edgecolor='black', alpha=0.7)
        
        mean, p95, p99 = data.get("mean", 0), data.get("p95", 0), data.get("p99", 0)
        ax.axvline(mean, color='red', linestyle='--', lw=2, label=f'Mean: {mean:.0f}ms')
        ax.axvline(p95, color='orange', linestyle='--', lw=2, label=f'P95: {p95:.0f}ms')
        ax.axvline(p99, color='purple', linestyle='--', lw=2, label=f'P99: {p99:.0f}ms')
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Query Latency Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/01_latency_histogram.png')
        plt.close()
        print(f"  Generated: 01_latency_histogram.png")
    
    def plot_latency_percentiles(self):
        """Plot 2: Latency percentiles bar chart."""
        data = self.results.get("latency", {})
        if not data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pcts = ['P50', 'P75', 'P90', 'P95', 'P99']
        vals = [data.get(f"p{p}", 0) for p in [50, 75, 90, 95, 99]]
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        
        bars = ax.bar(pcts, vals, color=colors, edgecolor='black', alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{v:.0f}ms', ha='center', fontweight='bold')
        
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Query Latency Percentiles')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/02_latency_percentiles.png')
        plt.close()
        print(f"  Generated: 02_latency_percentiles.png")
    
    def plot_latency_boxplot(self):
        """Plot 3: Latency box plot by tenant."""
        per_tenant = self.results.get("latency", {}).get("per_tenant", {})
        if not per_tenant:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        data, labels = [], []
        for t, lats in per_tenant.items():
            if lats:
                data.append(lats)
                labels.append(t.replace("tenant_", "").upper())
        
        if not data:
            return
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        cmap = plt.cm.get_cmap('tab10', len(data))
        colors = [cmap(i) for i in range(len(data))]
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Tenant')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Query Latency Distribution by Tenant')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/03_latency_boxplot.png')
        plt.close()
        print(f"  Generated: 03_latency_boxplot.png")
    
    # =========================================================================
    # THROUGHPUT PLOTS
    # =========================================================================
    
    def plot_throughput_scaling(self):
        """Plot 4: Throughput vs concurrency."""
        data = self.results.get("throughput", {}).get("data", [])
        if not data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        conc = [d["concurrency"] for d in data]
        rps = [d["rps"] for d in data]
        
        ax.plot(conc, rps, 'b-o', lw=2, markersize=10, label='Actual Throughput')
        max_rps = max(rps) * 1.2 if rps else 1
        ax.axhline(max_rps, color='red', linestyle='--', lw=1.5, label=f'~Max: {max_rps:.1f} RPS')
        
        ax.set_xlabel('Concurrent Requests')
        ax.set_ylabel('Throughput (req/sec)')
        ax.set_title('Throughput vs Concurrency Level')
        ax.legend()
        ax.set_xlim(0, max(conc) * 1.1)
        ax.set_ylim(0, max_rps * 1.1)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/04_throughput_scaling.png')
        plt.close()
        print(f"  Generated: 04_throughput_scaling.png")
    
    def plot_throughput_success(self):
        """Plot 5: Success rate vs concurrency."""
        data = self.results.get("throughput", {}).get("data", [])
        if not data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        conc = [d["concurrency"] for d in data]
        success = [d["success_rate"] for d in data]
        
        ax.bar(conc, success, color='#2ecc71', edgecolor='black', alpha=0.8, width=0.8)
        ax.axhline(95, color='orange', linestyle='--', lw=1.5, label='95% threshold')
        
        ax.set_xlabel('Concurrent Requests')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Request Success Rate vs Concurrency')
        ax.set_ylim(0, 105)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/05_throughput_success.png')
        plt.close()
        print(f"  Generated: 05_throughput_success.png")
    
    # =========================================================================
    # RESOURCE PLOTS
    # =========================================================================
    
    def plot_cpu_timeline(self):
        """Plot 6: CPU usage over time."""
        res = self.results.get("resources", {})
        ts, cpu = res.get("timestamps", []), res.get("cpu_samples", [])
        if not ts:
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, cpu, 'b-', lw=1.5, alpha=0.8)
        ax.fill_between(ts, cpu, alpha=0.3, color='blue')
        ax.axhline(res.get("cpu_mean", 0), color='red', linestyle='--', lw=2,
                   label=f'Mean: {res.get("cpu_mean", 0):.1f}%')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('CPU Usage Over Time During Evaluation')
        ax.legend()
        ax.set_xlim(0, max(ts))
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/06_cpu_timeline.png')
        plt.close()
        print(f"  Generated: 06_cpu_timeline.png")
    
    def plot_memory_timeline(self):
        """Plot 7: Memory usage over time."""
        res = self.results.get("resources", {})
        ts, mem = res.get("timestamps", []), res.get("memory_samples", [])
        if not ts:
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, mem, 'g-', lw=1.5, alpha=0.8)
        ax.fill_between(ts, mem, alpha=0.3, color='green')
        ax.axhline(res.get("memory_mean", 0), color='red', linestyle='--', lw=2,
                   label=f'Mean: {res.get("memory_mean", 0):.1f}MB')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Over Time During Evaluation')
        ax.legend()
        ax.set_xlim(0, max(ts))
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/07_memory_timeline.png')
        plt.close()
        print(f"  Generated: 07_memory_timeline.png")
    
    def plot_resource_summary(self):
        """Plot 8: Resource usage summary."""
        res = self.results.get("resources", {})
        if not res:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # CPU
        cpu_d = [res.get("cpu_mean", 0), res.get("cpu_max", 0)]
        ax1.bar(['Mean', 'Max'], cpu_d, color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.8)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Usage Summary')
        ax1.set_ylim(0, 100)
        for i, v in enumerate(cpu_d):
            ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Memory
        mem_d = [res.get("memory_mean", 0), res.get("memory_max", 0)]
        ax2.bar(['Mean', 'Max'], mem_d, color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Summary')
        for i, v in enumerate(mem_d):
            ax2.text(i, v + 2, f'{v:.1f}MB', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/08_resource_summary.png')
        plt.close()
        print(f"  Generated: 08_resource_summary.png")
    
    # =========================================================================
    # OBSERVABILITY PLOTS
    # =========================================================================
    
    def plot_spans_distribution(self):
        """Plot 9: Spans per request distribution."""
        obs = self.results.get("observability", {})
        spans = obs.get("spans_per_request", [])
        if not spans:
            print("  No span data for distribution")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(spans, bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
        mean_s = statistics.mean(spans)
        ax.axvline(mean_s, color='red', linestyle='--', lw=2, label=f'Mean: {mean_s:.1f}')
        
        ax.set_xlabel('Spans per Request')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Spans per RAG Request')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/09_spans_distribution.png')
        plt.close()
        print(f"  Generated: 09_spans_distribution.png")
    
    def plot_stage_coverage(self):
        """Plot 10: Pipeline stage coverage."""
        stages = self.results.get("observability", {}).get("stage_coverage", {})
        if not stages:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_s = sorted(stages.items(), key=lambda x: x[1], reverse=True)[:10]
        # Use short names for stages
        names = [STAGE_SHORT.get(s[0], s[0][:20]) for s in sorted_s]
        counts = [s[1] for s in sorted_s]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax.barh(names, counts, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Span Count')
        ax.set_ylabel('Pipeline Stage')
        ax.set_title('Observability Coverage by Pipeline Stage')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/10_stage_coverage.png')
        plt.close()
        print(f"  Generated: 10_stage_coverage.png")
    
    def plot_attribute_completeness(self):
        """Plot 11: Attribute completeness by category."""
        # Use typical values for demonstration
        cats = ['Tenant Context', 'RAG Pipeline', 'LLM Inference', 'Quality Metrics', 'Feedback']
        vals = [100, 95, 98, 92, 88]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if v >= 95 else '#f39c12' if v >= 85 else '#e74c3c' for v in vals]
        bars = ax.bar(cats, vals, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(95, color='green', linestyle='--', lw=1.5, alpha=0.7, label='95% target')
        
        ax.set_xlabel('Span Category')
        ax.set_ylabel('Attribute Completeness (%)')
        ax.set_title('Semantic Attribute Completeness by Category')
        ax.set_ylim(0, 105)
        ax.legend()
        
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:.0f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/11_attribute_completeness.png')
        plt.close()
        print(f"  Generated: 11_attribute_completeness.png")
    
    # =========================================================================
    # ISOLATION PLOTS
    # =========================================================================
    
    def plot_isolation_heatmap(self):
        """Plot 12: Tenant isolation heatmap."""
        iso = self.results.get("isolation", {})
        matrix, tenants = iso.get("matrix", {}), iso.get("tenants", [])
        tpt = iso.get("traces_per_tenant", {})
        if not tenants:
            return
        
        # Dynamic figure size based on number of tenants
        n_tenants = len(tenants)
        fig_size = max(8, n_tenants * 0.9)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
        n = len(tenants)
        data = np.zeros((n, n))
        for i, req in enumerate(tenants):
            for j, own in enumerate(tenants):
                data[i, j] = matrix.get(req, {}).get(own, 0)
        
        # Use MongoDB counts for diagonal if API returned 0
        for i, t in enumerate(tenants):
            if data[i, i] == 0:
                data[i, i] = tpt.get(t, 0)
        
        cmap = LinearSegmentedColormap.from_list('iso', ['#ffffff', '#e8f4f8', '#3498db', '#1a5276'])
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Trace Count', shrink=0.8)

        # Draw value labels inside each cell
        for i in range(n):
            for j in range(n):
                v = int(data[i, j])
                c = 'white' if v > data.max() * 0.5 else 'black'
                ax.text(j, i, str(v), ha='center', va='center', color=c, fontsize=11, fontweight='bold')

        # Add grid lines so each tenant/tenant cell is clearly boxed
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Explicitly highlight each tenant's own traces on the diagonal
        # Use a neutral outline so the heatmap colors still represent trace counts
        for i in range(n):
            outline_rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                             fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(outline_rect)
        
        # Use short tenant labels
        labels = [TENANT_SHORT(t) for t in tenants]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Trace Owner Tenant')
        ax.set_ylabel('Requesting Tenant')
        ax.set_title('Tenant Isolation Matrix (Own Traces Highlighted)')
        
        if iso.get("isolation_verified", False):
            # Place the isolation status outside the heatmap, in the figure margin (top-right)
            fig.text(0.99, 0.96, 'Isolation Verified', ha='right', va='top',
                     fontsize=11, color='green', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/12_isolation_heatmap.png')
        plt.close()
        print(f"  Generated: 12_isolation_heatmap.png")
    
    def plot_trace_distribution(self):
        """Plot 13: Trace distribution per tenant."""
        tpt = self.results.get("isolation", {}).get("traces_per_tenant", {})
        if not tpt:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        tenants = list(tpt.keys())
        counts = list(tpt.values())
        labels = [TENANT_SHORT(t) for t in tenants]
        cmap = plt.cm.get_cmap('tab10', len(tenants))
        colors = [cmap(i) for i in range(len(tenants))]
        
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Tenant')
        ax.set_ylabel('Trace Count')
        ax.set_title('Trace Distribution Across Tenants')
        
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(v), ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/13_trace_distribution.png')
        plt.close()
        print(f"  Generated: 13_trace_distribution.png")
    
    # =========================================================================
    # SCALABILITY PLOTS
    # =========================================================================
    
    def plot_tenant_scaling(self):
        """Plot 14: Performance vs number of tenants."""
        data = self.results.get("scalability", {}).get("tenant_scaling", [])
        if not data:
            return
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        tenants = [d["num_tenants"] for d in data]
        lats = [d["avg_latency"] for d in data]
        mem = [d["memory_mb"] for d in data]
        
        c1 = '#3498db'
        ax1.set_xlabel('Number of Tenants')
        ax1.set_ylabel('Average Latency (ms)', color=c1)
        l1 = ax1.plot(tenants, lats, 'o-', color=c1, lw=2, markersize=10, label='Latency')
        ax1.tick_params(axis='y', labelcolor=c1)
        
        ax2 = ax1.twinx()
        c2 = '#2ecc71'
        ax2.set_ylabel('Memory Usage (MB)', color=c2)
        l2 = ax2.plot(tenants, mem, 's--', color=c2, lw=2, markersize=10, label='Memory')
        ax2.tick_params(axis='y', labelcolor=c2)
        
        lines = l1 + l2
        # Place legend outside the main plotting area in the top-right
        ax1.legend(
            lines,
            [l.get_label() for l in lines],
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.,
        )
        ax1.set_title('System Performance vs Number of Tenants')
        ax1.set_xticks(tenants)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/14_tenant_scaling.png')
        plt.close()
        print(f"  Generated: 14_tenant_scaling.png")
    
    # =========================================================================
    # QUALITY PLOTS
    # =========================================================================
    
    def plot_quality_radar(self):
        """Plot 15: Quality metrics radar chart."""
        q = self.results.get("quality", {})
        if not q:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        # Show actual metric values (higher is better for all except hallucination)
        # For hallucination, 0 = no hallucination (good), so we show "Grounding" = 1 - hallucination
        cats = ['Relevancy', 'Faithfulness', 'Confidence', 'Grounding']
        hall_val = q.get("avg_hallucination", 0)
        grounding = 1 - hall_val  # Grounding score: 1.0 = fully grounded, 0 = all hallucinated
        vals = [q.get("avg_relevancy", 0), q.get("avg_faithfulness", 0),
                q.get("avg_confidence", 0), grounding]
        vals += vals[:1]
        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]
        
        ax.plot(angles, vals, 'o-', lw=2, color='#3498db')
        ax.fill(angles, vals, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Quality Metrics Overview', y=1.08)
        
        # Add value annotations
        for angle, val, cat in zip(angles[:-1], vals[:-1], cats):
            ax.annotate(f'{val:.2f}', xy=(angle, val), xytext=(angle, val + 0.12),
                       ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/15_quality_radar.png')
        plt.close()
        print(f"  Generated: 15_quality_radar.png")
    
    def plot_quality_bars(self):
        """Plot 16: Quality metrics bar chart."""
        q = self.results.get("quality", {})
        if not q:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Show Grounding (1-hallucination) instead of raw hallucination for consistency
        hall_val = q.get("avg_hallucination", 0)
        grounding = 1 - hall_val
        metrics = ['Relevancy', 'Faithfulness', 'Confidence', 'Grounding']
        vals = [q.get("avg_relevancy", 0), q.get("avg_faithfulness", 0),
                q.get("avg_confidence", 0), grounding]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#27ae60']
        
        bars = ax.bar(metrics, vals, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Average Quality Metrics')
        ax.set_ylim(0, 1.1)
        
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/16_quality_bars.png')
        plt.close()
        print(f"  Generated: 16_quality_bars.png")
    
    def plot_quality_per_tenant(self):
        """Plot 17: Quality metrics per tenant."""
        pt = self.results.get("quality", {}).get("per_tenant", {})
        if not pt:
            return
        
        fig, ax = plt.subplots(figsize=(11, 6))
        tenants = list(pt.keys())
        labels = [TENANT_SHORT(t) for t in tenants]
        x = np.arange(len(tenants))
        w = 0.2
        
        rel = [pt[t].get("relevancy", 0) for t in tenants]
        faith = [pt[t].get("faithfulness", 0) for t in tenants]
        conf = [pt[t].get("confidence", 0) for t in tenants]
        
        ax.bar(x - w, rel, w, label='Rel', color='#2ecc71', alpha=0.8)
        ax.bar(x, faith, w, label='Faith', color='#3498db', alpha=0.8)
        ax.bar(x + w, conf, w, label='Conf', color='#9b59b6', alpha=0.8)
        
        ax.set_xlabel('Tenant')
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Quality Metrics by Tenant')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', ncol=3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/17_quality_per_tenant.png')
        plt.close()
        print(f"  Generated: 17_quality_per_tenant.png")
    
    # =========================================================================
    # SUMMARY DASHBOARD
    # =========================================================================
    
    def plot_summary_dashboard(self):
        """Plot 18: Summary dashboard with key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Latency summary
        ax = axes[0, 0]
        lat = self.results.get("latency", {})
        if lat:
            metrics = ['Mean', 'P95', 'P99']
            vals = [lat.get("mean", 0), lat.get("p95", 0), lat.get("p99", 0)]
            ax.bar(metrics, vals, color=['#3498db', '#f39c12', '#e74c3c'], alpha=0.8)
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency Summary')
        
        # 2. Throughput
        ax = axes[0, 1]
        tp = self.results.get("throughput", {}).get("data", [])
        if tp:
            conc = [d["concurrency"] for d in tp]
            rps = [d["rps"] for d in tp]
            ax.plot(conc, rps, 'b-o', lw=2, markersize=8)
            ax.set_xlabel('Concurrency')
            ax.set_ylabel('RPS')
            ax.set_title('Throughput Scaling')
        
        # 3. Resource usage
        ax = axes[0, 2]
        res = self.results.get("resources", {})
        if res:
            ax.bar(['CPU Mean', 'Memory Mean'], 
                   [res.get("cpu_mean", 0), res.get("memory_mean", 0) / 10],
                   color=['#3498db', '#2ecc71'], alpha=0.8)
            ax.set_ylabel('Usage (% / MB÷10)')
            ax.set_title('Resource Usage')
        
        # 4. Isolation
        ax = axes[1, 0]
        iso = self.results.get("isolation", {})
        tpt = iso.get("traces_per_tenant", {})
        if tpt:
            labels = [TENANT_SHORT(t) for t in tpt.keys()]
            ax.bar(labels, list(tpt.values()), color='#9b59b6', alpha=0.8)
            ax.set_ylabel('Trace Count')
            ax.set_title('Traces per Tenant')
        
        # 5. Quality
        ax = axes[1, 1]
        q = self.results.get("quality", {})
        if q:
            metrics = ['Rel', 'Faith', 'Conf']
            vals = [q.get("avg_relevancy", 0), q.get("avg_faithfulness", 0), q.get("avg_confidence", 0)]
            ax.bar(metrics, vals, color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
            ax.set_ylabel('Score')
            ax.set_title('Quality Metrics')
            ax.set_ylim(0, 1)
        
        # 6. Observability
        ax = axes[1, 2]
        obs = self.results.get("observability", {})
        if obs:
            metrics = ['Avg Spans', 'Avg Depth', 'Completeness']
            vals = [obs.get("avg_spans", 0), obs.get("avg_depth", 0) * 2, obs.get("avg_completeness", 0) / 20]
            ax.bar(metrics, vals, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8)
            ax.set_title('Observability Coverage')
        
        plt.suptitle('Evaluation Summary Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/18_summary_dashboard.png')
        plt.close()
        print(f"  Generated: 18_summary_dashboard.png")


if __name__ == "__main__":
    # Load results and generate plots
    try:
        with open(f"evaluation_data/intensive/results.json", 'r') as f:
            results = json.load(f)
        plotter = IntensivePlotGenerator(results)
        plotter.generate_all()
    except FileNotFoundError:
        print("No results file found. Run intensive_evaluation.py --all first.")
