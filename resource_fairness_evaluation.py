#!/usr/bin/env python3
"""
Resource Fairness Evaluation.

Tests that the resource quota system ensures fair distribution across tenants
and prevents a single tenant from consuming all resources.

Evaluation Scenarios:
1. Single tenant burst - verify rate limiting kicks in
2. Multi-tenant concurrent load - verify fair distribution
3. Greedy tenant scenario - verify other tenants still get resources
4. Fairness index calculation
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Any

import requests
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:8001")
API_V1 = f"{BASE_URL}/api/v1"

TENANTS = [
    {"username": "tenant_a", "password": "112233"},
    {"username": "tenant_b", "password": "112233"},
    {"username": "tenant_c", "password": "112233"},
    {"username": "tenant_d", "password": "112233"},
    {"username": "tenant_e", "password": "112233"},
    {"username": "tenant_f", "password": "112233"},
    {"username": "tenant_g", "password": "112233"},
    {"username": "tenant_h", "password": "112233"},
    {"username": "tenant_i", "password": "112233"},
    {"username": "tenant_j", "password": "112233"},
]

TEST_QUERIES = [
    # India trips
    "Trips to India",
    "Cheapest trips to India",
    "Best time to visit India",
    "Cultural tours in India",
    "Budget travel to India",
    # Italy trips
    "Cheapest trips to Italy",
    "Earliest trips to Italy",
    "Romantic getaways in Italy",
    "Best food tours in Italy",
    "Luxury vacations in Italy",
    # Spain trips
    "Beach vacations in Spain",
    "Cheapest trips to Spain",
    "Adventure trips to Spain",
    "Barcelona city breaks",
    "Madrid cultural tours",
    # France trips
    "Paris city breaks",
    "Wine tours in France",
    "Cheapest trips to France",
    "French Riviera vacations",
    "Loire Valley castles tour",
    # Greece trips
    "Island hopping in Greece",
    "Cheapest trips to Greece",
    "Santorini romantic getaways",
    "Athens ancient ruins tour",
    # Thailand trips
    "Beach resorts in Thailand",
    "Cheapest trips to Thailand",
    "Bangkok city tours",
    "Phuket beach vacations",
    # Japan trips
    "Cultural trips to Japan",
    "Cheapest trips to Japan",
    "Cherry blossom tours Japan",
    "Tokyo city exploration",
    # General travel queries
    "Best family vacation destinations",
    "Cheapest European destinations",
    "Adventure travel packages",
    "Luxury spa retreats",
    "Budget backpacking trips",
    "Solo travel destinations",
    "Group tour packages",
    "Honeymoon destinations",
    "Safari trips to Africa",
    "Cruise vacation deals",
]

PLOTS_DIR = "plots/resource_fairness"
DATA_DIR = "evaluation_data/resource_fairness"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})


class ResourceFairnessEvaluator:
    """Evaluates resource fairness across tenants."""
    
    def __init__(self):
        self.session = requests.Session()
        self.tenant_tokens = {}
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "single_tenant_burst": {},
            "multi_tenant_load": {},
            "greedy_tenant": {},
            "fairness_metrics": {},
            "quota_stats": {},
        }
        self._authenticate_tenants()
    
    def _authenticate_tenants(self):
        print("=" * 60)
        print("AUTHENTICATING TENANTS")
        print("=" * 60)
        for tenant in TENANTS:
            try:
                resp = self.session.post(f"{API_V1}/auth/login",
                    data={"username": tenant["username"], "password": tenant["password"]}, timeout=30)
                if resp.status_code == 200:
                    token = resp.json().get("access_token")
                    if token:
                        self.tenant_tokens[tenant["username"]] = token
                        print(f"  ✓ {tenant['username']} authenticated")
            except Exception as e:
                print(f"  ✗ {tenant['username']} - {e}")
        print()
    
    def _refresh_token(self, tenant_id: str) -> bool:
        tenant = next((t for t in TENANTS if t["username"] == tenant_id), None)
        if not tenant:
            return False
        try:
            resp = self.session.post(f"{API_V1}/auth/login",
                data={"username": tenant["username"], "password": tenant["password"]}, timeout=30)
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                if token:
                    self.tenant_tokens[tenant_id] = token
                    return True
        except:
            pass
        return False
    
    def _get_headers(self, tenant_id: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "x-tenant-id": tenant_id}
        if tenant_id in self.tenant_tokens:
            headers["Authorization"] = f"Bearer {self.tenant_tokens[tenant_id]}"
        return headers
    
    def _run_query(self, query: str, tenant_id: str, timeout: int = 120) -> Dict:
        start = time.time()
        result = {"tenant_id": tenant_id, "success": False, "latency_ms": 0, "rate_limited": False}
        try:
            resp = self.session.post(f"{API_V1}/rag/query",
                headers=self._get_headers(tenant_id),
                json={"query": query, "top_k": 3,
                      "conversation_id": f"fairness-{tenant_id}-{int(time.time()*1000)}"},
                timeout=timeout)
            
            if resp.status_code == 401:
                if self._refresh_token(tenant_id):
                    resp = self.session.post(f"{API_V1}/rag/query",
                        headers=self._get_headers(tenant_id),
                        json={"query": query, "top_k": 3,
                              "conversation_id": f"fairness-{tenant_id}-{int(time.time()*1000)}"},
                        timeout=timeout)
            
            result["latency_ms"] = (time.time() - start) * 1000
            result["status_code"] = resp.status_code
            
            if resp.status_code == 200:
                result["success"] = True
            elif resp.status_code == 429:
                result["rate_limited"] = True
                result["error"] = resp.json().get("detail", {}).get("error", "RATE_LIMITED")
            else:
                result["error"] = resp.text[:100]
        except Exception as e:
            result["latency_ms"] = (time.time() - start) * 1000
            result["error"] = str(e)
        return result
    
    def _get_quota_stats(self, tenant_id: str) -> Dict:
        try:
            resp = self.session.get(f"{API_V1}/observability/resource-quotas",
                headers=self._get_headers(tenant_id), timeout=30)
            if resp.status_code == 401:
                if self._refresh_token(tenant_id):
                    resp = self.session.get(f"{API_V1}/observability/resource-quotas",
                        headers=self._get_headers(tenant_id), timeout=30)
            return resp.json() if resp.status_code == 200 else {}
        except:
            return {}
    
    def evaluate_single_tenant_burst(self, num_requests: int = 20, concurrency: int = 10):
        """
        Scenario 1: Single tenant tries to send many CONCURRENT requests.
        Verify that concurrent limit and rate limiting kick in.
        
        With concurrent_limit=3 and 10 concurrent requests, 7 should be rejected.
        """
        print("=" * 60)
        print(f"SCENARIO 1: SINGLE TENANT BURST ({num_requests} requests, {concurrency} concurrent)")
        print("=" * 60)
        
        tenant = TENANTS[0]["username"]
        self._refresh_token(tenant)
        
        results = []
        
        print(f"  Sending {num_requests} concurrent requests from {tenant}...")
        print(f"  (Concurrent limit is 3, so excess requests should be rejected)")
        
        # Send requests concurrently to trigger concurrent limit
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_requests):
                query = TEST_QUERIES[i % len(TEST_QUERIES)]
                futures.append(executor.submit(self._run_query, query, tenant))
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else ("⚠ LIMITED" if result["rate_limited"] else "✗")
                print(f"    Request {i+1}/{num_requests}: {status} ({result['latency_ms']:.0f}ms)")
        
        successful = sum(1 for r in results if r["success"])
        rate_limited_count = sum(1 for r in results if r.get("rate_limited"))
        latencies = [r["latency_ms"] for r in results if r["success"]]
        
        self.results["single_tenant_burst"] = {
            "tenant": tenant,
            "total_requests": num_requests,
            "concurrency": concurrency,
            "successful": successful,
            "rate_limited": rate_limited_count,
            "success_rate": successful / num_requests * 100,
            "rate_limit_rate": rate_limited_count / num_requests * 100,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "results": results,
        }
        
        print(f"\n  Results: {successful} successful, {rate_limited_count} rate limited")
        print(f"  Rate limiting effectiveness: {rate_limited_count / num_requests * 100:.1f}%")
        return self.results["single_tenant_burst"]
    
    def evaluate_multi_tenant_load(self, requests_per_tenant: int = 10):
        """
        Scenario 2: Multiple tenants sending requests concurrently.
        Verify fair distribution of resources.
        """
        print("\n" + "=" * 60)
        print(f"SCENARIO 2: MULTI-TENANT CONCURRENT LOAD")
        print("=" * 60)
        
        # Refresh all tokens
        for t in TENANTS:
            self._refresh_token(t["username"])
        
        per_tenant_results = {t["username"]: [] for t in TENANTS}
        
        def run_tenant_requests(tenant_id: str, num_requests: int):
            results = []
            for i in range(num_requests):
                query = TEST_QUERIES[i % len(TEST_QUERIES)]
                result = self._run_query(query, tenant_id)
                results.append(result)
                time.sleep(0.2)  # Small delay between requests
            return tenant_id, results
        
        print(f"  Running {requests_per_tenant} requests per tenant concurrently...")
        
        with ThreadPoolExecutor(max_workers=len(TENANTS)) as executor:
            futures = [executor.submit(run_tenant_requests, t["username"], requests_per_tenant) 
                      for t in TENANTS]
            
            for future in as_completed(futures):
                tenant_id, results = future.result()
                per_tenant_results[tenant_id] = results
                successful = sum(1 for r in results if r["success"])
                print(f"    {tenant_id}: {successful}/{len(results)} successful")
        
        # Calculate fairness metrics
        success_counts = [sum(1 for r in results if r["success"]) for results in per_tenant_results.values()]
        latencies_per_tenant = {
            tid: [r["latency_ms"] for r in results if r["success"]]
            for tid, results in per_tenant_results.items()
        }
        
        # Jain's fairness index
        n = len(success_counts)
        sum_x = sum(success_counts)
        sum_x2 = sum(x**2 for x in success_counts)
        fairness_index = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0
        
        self.results["multi_tenant_load"] = {
            "requests_per_tenant": requests_per_tenant,
            "success_per_tenant": {tid: sum(1 for r in results if r["success"]) 
                                   for tid, results in per_tenant_results.items()},
            "rate_limited_per_tenant": {tid: sum(1 for r in results if r.get("rate_limited")) 
                                        for tid, results in per_tenant_results.items()},
            "avg_latency_per_tenant": {tid: statistics.mean(lats) if lats else 0 
                                       for tid, lats in latencies_per_tenant.items()},
            "fairness_index": fairness_index,
            "total_successful": sum(success_counts),
            "total_requests": requests_per_tenant * len(TENANTS),
        }
        
        print(f"\n  Fairness Index: {fairness_index:.4f} (1.0 = perfectly fair)")
        return self.results["multi_tenant_load"]
    
    def evaluate_greedy_tenant(self, greedy_requests: int = 15, greedy_concurrency: int = 10, other_requests: int = 3):
        """
        Scenario 3: One tenant tries to be greedy with CONCURRENT requests while others need resources.
        Verify that greedy tenant is throttled and others still get service.
        
        Greedy tenant sends 15 requests with 10 concurrent (exceeds limit of 3).
        Other tenants send 3 requests each sequentially.
        """
        print("\n" + "=" * 60)
        print(f"SCENARIO 3: GREEDY TENANT SCENARIO")
        print("=" * 60)
        
        # Refresh all tokens
        for t in TENANTS:
            self._refresh_token(t["username"])
        
        greedy_tenant = TENANTS[0]["username"]
        other_tenants = [t["username"] for t in TENANTS[1:]]
        
        results = {"greedy": [], "others": {t: [] for t in other_tenants}}
        
        def greedy_worker():
            """Greedy tenant sends many concurrent requests."""
            with ThreadPoolExecutor(max_workers=greedy_concurrency) as ex:
                futures = [ex.submit(self._run_query, TEST_QUERIES[i % len(TEST_QUERIES)], greedy_tenant)
                          for i in range(greedy_requests)]
                for f in as_completed(futures):
                    results["greedy"].append(f.result())
        
        def other_worker(tenant_id: str):
            """Other tenants send requests at normal pace."""
            for i in range(other_requests):
                query = TEST_QUERIES[i % len(TEST_QUERIES)]
                result = self._run_query(query, tenant_id)
                results["others"][tenant_id].append(result)
                time.sleep(1)  # Normal pace
        
        print(f"  Greedy tenant ({greedy_tenant}): {greedy_requests} requests with {greedy_concurrency} concurrent")
        print(f"  Other tenants: {other_requests} requests each at normal pace")
        print(f"  (Concurrent limit is 3, so greedy tenant should be throttled)")
        
        with ThreadPoolExecutor(max_workers=len(TENANTS)) as executor:
            futures = [executor.submit(greedy_worker)]
            for tid in other_tenants:
                futures.append(executor.submit(other_worker, tid))
            
            for future in as_completed(futures):
                future.result()
        
        # Analyze results
        greedy_success = sum(1 for r in results["greedy"] if r["success"])
        greedy_limited = sum(1 for r in results["greedy"] if r.get("rate_limited"))
        
        other_success = {tid: sum(1 for r in res if r["success"]) 
                        for tid, res in results["others"].items()}
        
        self.results["greedy_tenant"] = {
            "greedy_tenant": greedy_tenant,
            "greedy_requests": greedy_requests,
            "greedy_concurrency": greedy_concurrency,
            "greedy_successful": greedy_success,
            "greedy_rate_limited": greedy_limited,
            "greedy_throttle_rate": greedy_limited / greedy_requests * 100 if greedy_requests > 0 else 0,
            "other_tenants_success": other_success,
            "other_tenants_total": sum(other_success.values()),
            "other_tenants_expected": other_requests * len(other_tenants),
            "protection_effective": greedy_limited > 0 and sum(other_success.values()) > 0,
        }
        
        print(f"\n  Greedy tenant: {greedy_success}/{greedy_requests} successful, {greedy_limited} throttled")
        print(f"  Other tenants: {sum(other_success.values())}/{other_requests * len(other_tenants)} successful")
        print(f"  Protection effective: {'YES' if self.results['greedy_tenant']['protection_effective'] else 'NO'}")
        
        return self.results["greedy_tenant"]
    
    def collect_quota_stats(self):
        """Collect final quota statistics."""
        print("\n" + "=" * 60)
        print("COLLECTING QUOTA STATISTICS")
        print("=" * 60)
        
        self._refresh_token(TENANTS[0]["username"])
        stats = self._get_quota_stats(TENANTS[0]["username"])
        self.results["quota_stats"] = stats
        
        if stats.get("fairness"):
            print(f"  Fairness Index: {stats['fairness'].get('fairness_index', 0):.4f}")
        if stats.get("all_tenants"):
            for tid, data in stats["all_tenants"].items():
                print(f"  {tid}: {data.get('total_requests', 0)} requests, {data.get('rejected_requests', 0)} rejected")
        
        return stats
    
    def save_results(self):
        """Save results to JSON."""
        filepath = f"{DATA_DIR}/results.json"
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")
    
    def generate_plots(self):
        """Generate visualization plots."""
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        self._plot_single_tenant_burst()
        self._plot_multi_tenant_distribution()
        self._plot_greedy_tenant_protection()
        self._plot_fairness_summary()
        
        print(f"\nPlots saved to: {PLOTS_DIR}/")
    
    def _plot_single_tenant_burst(self):
        """Plot single tenant burst results."""
        data = self.results.get("single_tenant_burst", {})
        if not data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Request outcomes
        outcomes = ['Successful', 'Rate Limited', 'Failed']
        counts = [
            data.get("successful", 0),
            data.get("rate_limited", 0),
            data.get("total_requests", 0) - data.get("successful", 0) - data.get("rate_limited", 0)
        ]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax1.bar(outcomes, counts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_ylabel('Request Count')
        ax1.set_title(f'Single Tenant Burst Test ({data.get("total_requests", 0)} requests)')
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # Latency over time
        results = data.get("results", [])
        if results:
            latencies = [r["latency_ms"] for r in results]
            colors_lat = ['#2ecc71' if r["success"] else '#f39c12' if r.get("rate_limited") else '#e74c3c' 
                         for r in results]
            ax2.scatter(range(len(latencies)), latencies, c=colors_lat, alpha=0.7, s=50)
            ax2.set_xlabel('Request Number')
            ax2.set_ylabel('Latency (ms)')
            ax2.set_title('Request Latency Over Time')
            ax2.axhline(statistics.mean([l for l, r in zip(latencies, results) if r["success"]]) if any(r["success"] for r in results) else 0,
                       color='blue', linestyle='--', label='Avg Successful')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/01_single_tenant_burst.png')
        plt.close()
        print(f"  Generated: 01_single_tenant_burst.png")
    
    def _plot_multi_tenant_distribution(self):
        """Plot multi-tenant load distribution."""
        data = self.results.get("multi_tenant_load", {})
        if not data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Success per tenant
        success = data.get("success_per_tenant", {})
        if success:
            tenants = list(success.keys())
            counts = list(success.values())
            labels = [t.replace("tenant_", "").upper() for t in tenants]
            # Generate enough colors for all tenants
            cmap = plt.cm.get_cmap('tab10', len(tenants))
            colors = [cmap(i) for i in range(len(tenants))]
            
            ax1.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
            ax1.axhline(statistics.mean(counts), color='red', linestyle='--', lw=2, label='Mean')
            ax1.set_ylabel('Successful Requests')
            ax1.set_xlabel('Tenant')
            ax1.set_title('Request Success Distribution')
            ax1.legend()
            
            for i, v in enumerate(counts):
                ax1.text(i, v + 0.2, str(v), ha='center', fontweight='bold')
        
        # Latency per tenant
        latencies = data.get("avg_latency_per_tenant", {})
        if latencies:
            tenants = list(latencies.keys())
            lats = list(latencies.values())
            labels = [t.replace("tenant_", "").upper() for t in tenants]
            
            cmap2 = plt.cm.get_cmap('tab10', len(tenants))
            colors2 = [cmap2(i) for i in range(len(tenants))]
            ax2.bar(labels, lats, color=colors2, edgecolor='black', alpha=0.8)
            ax2.set_ylabel('Average Latency (ms)')
            ax2.set_xlabel('Tenant')
            ax2.set_title('Latency Distribution Across Tenants')
        
        # Add fairness index annotation
        fairness = data.get("fairness_index", 0)
        fig.suptitle(f'Multi-Tenant Load Test (Fairness Index: {fairness:.4f})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/02_multi_tenant_distribution.png')
        plt.close()
        print(f"  Generated: 02_multi_tenant_distribution.png")
    
    def _plot_greedy_tenant_protection(self):
        """Plot greedy tenant protection results."""
        data = self.results.get("greedy_tenant", {})
        if not data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Greedy tenant outcomes
        greedy_data = [data.get("greedy_successful", 0), data.get("greedy_rate_limited", 0)]
        ax1.pie(greedy_data, labels=['Successful', 'Rate Limited'], 
               colors=['#2ecc71', '#f39c12'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Greedy Tenant ({data.get("greedy_tenant", "")}) Outcomes')
        
        # Other tenants success
        other_success = data.get("other_tenants_success", {})
        if other_success:
            tenants = list(other_success.keys())
            counts = list(other_success.values())
            labels = [t.replace("tenant_", "").upper() for t in tenants]
            cmap = plt.cm.get_cmap('tab10', len(tenants))
            colors = [cmap(i) for i in range(len(tenants))]
            
            ax2.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
            ax2.set_ylabel('Successful Requests')
            ax2.set_xlabel('Tenant')
            ax2.set_title('Other Tenants Success (While Greedy Tenant Active)')
            
            for i, v in enumerate(counts):
                ax2.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        protection = "EFFECTIVE" if data.get("protection_effective") else "NOT EFFECTIVE"
        fig.suptitle(f'Greedy Tenant Protection Test - {protection}', fontsize=14, fontweight='bold',
                    color='green' if data.get("protection_effective") else 'red')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/03_greedy_tenant_protection.png')
        plt.close()
        print(f"  Generated: 03_greedy_tenant_protection.png")
    
    def _plot_fairness_summary(self):
        """Plot overall fairness summary."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Fairness index comparison
        ax = axes[0, 0]
        scenarios = ['Multi-Tenant\nLoad', 'Greedy\nScenario']
        indices = [
            self.results.get("multi_tenant_load", {}).get("fairness_index", 0),
            1.0 if self.results.get("greedy_tenant", {}).get("protection_effective") else 0.5,
        ]
        colors = ['#2ecc71' if i >= 0.9 else '#f39c12' if i >= 0.7 else '#e74c3c' for i in indices]
        ax.bar(scenarios, indices, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(0.9, color='green', linestyle='--', lw=2, label='Good (0.9)')
        ax.axhline(0.7, color='orange', linestyle='--', lw=2, label='Fair (0.7)')
        ax.set_ylabel('Fairness Index')
        ax.set_title('Fairness Index by Scenario')
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        # 2. Rate limiting effectiveness
        ax = axes[0, 1]
        burst = self.results.get("single_tenant_burst", {})
        greedy = self.results.get("greedy_tenant", {})
        scenarios = ['Single Burst', 'Greedy Tenant']
        throttle_rates = [
            burst.get("rate_limit_rate", 0),
            greedy.get("greedy_throttle_rate", 0),
        ]
        ax.bar(scenarios, throttle_rates, color=['#f39c12', '#e74c3c'], edgecolor='black', alpha=0.8)
        ax.set_ylabel('Throttle Rate (%)')
        ax.set_title('Rate Limiting Effectiveness')
        ax.set_ylim(0, 100)
        
        # 3. Resource distribution pie
        ax = axes[1, 0]
        multi = self.results.get("multi_tenant_load", {})
        success = multi.get("success_per_tenant", {})
        if success:
            labels = [t.replace("tenant_", "").upper() for t in success.keys()]
            sizes = list(success.values())
            cmap = plt.cm.get_cmap('tab10', len(labels))
            colors = [cmap(i) for i in range(len(labels))]
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Resource Distribution Across Tenants')
        
        # 4. Summary metrics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        RESOURCE FAIRNESS EVALUATION SUMMARY
        =====================================
        
        Single Tenant Burst:
          • Requests: {burst.get('total_requests', 0)}
          • Rate Limited: {burst.get('rate_limited', 0)} ({burst.get('rate_limit_rate', 0):.1f}%)
        
        Multi-Tenant Load:
          • Fairness Index: {multi.get('fairness_index', 0):.4f}
          • Total Successful: {multi.get('total_successful', 0)}/{multi.get('total_requests', 0)}
        
        Greedy Tenant Protection:
          • Greedy Throttled: {greedy.get('greedy_rate_limited', 0)}
          • Others Served: {greedy.get('other_tenants_total', 0)}/{greedy.get('other_tenants_expected', 0)}
          • Protection: {'✓ EFFECTIVE' if greedy.get('protection_effective') else '✗ NOT EFFECTIVE'}
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Resource Fairness Evaluation Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/04_fairness_summary.png')
        plt.close()
        print(f"  Generated: 04_fairness_summary.png")
    
    def run_all(self):
        """Run all evaluation scenarios."""
        print("\n" + "=" * 60)
        print("RESOURCE FAIRNESS EVALUATION")
        print("=" * 60 + "\n")
        
        # Scenario 1: Single tenant burst with concurrent requests
        # 20 requests with 10 concurrent, limit is 3, so ~7 should be rejected
        self.evaluate_single_tenant_burst(num_requests=20, concurrency=10)
        
        # Scenario 2: Multi-tenant concurrent load
        self.evaluate_multi_tenant_load(requests_per_tenant=5)
        
        # Scenario 3: Greedy tenant with concurrent requests
        # 15 requests with 10 concurrent, limit is 3
        self.evaluate_greedy_tenant(greedy_requests=15, greedy_concurrency=10, other_requests=3)
        
        self.collect_quota_stats()
        self.save_results()
        self.generate_plots()
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resource Fairness Evaluation")
    parser.add_argument("--burst", type=int, default=40, help="Requests for burst test")
    parser.add_argument("--load", type=int, default=10, help="Requests per tenant for load test")
    parser.add_argument("--greedy", type=int, default=30, help="Greedy tenant requests")
    args = parser.parse_args()
    
    evaluator = ResourceFairnessEvaluator()
    evaluator.run_all()
