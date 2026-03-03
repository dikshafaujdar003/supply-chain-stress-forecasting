import networkx as nx
import pandas as pd

SUPPLY_CHAIN_EDGES = [
    ("TSM",  "NVDA",  "semiconductor_foundry",   "semiconductors"),
    ("TSM",  "AAPL",  "semiconductor_foundry",   "semiconductors"),
    ("ASML", "TSM",   "chip_equipment",          "semiconductors"),
    ("AMAT", "TSM",   "chip_equipment",          "semiconductors"),
    ("MU",   "NVDA",  "memory_supplier",         "semiconductors"),
    ("MU",   "AAPL",  "memory_supplier",         "semiconductors"),
    ("HON",  "GE",    "components_supplier",     "manufacturing"),
    ("HON",  "F",     "components_supplier",     "automotive"),
    ("EMR",  "HON",   "industrial_components",   "manufacturing"),
    ("AA",   "F",     "aluminum_supplier",       "automotive"),
    ("AA",   "GM",    "aluminum_supplier",       "automotive"),
    ("FCX",  "HON",   "copper_supplier",         "manufacturing"),
    ("FCX",  "NVDA",  "copper_supplier",         "semiconductors"),
    ("NUE",  "F",     "steel_supplier",          "automotive"),
    ("NUE",  "GM",    "steel_supplier",          "automotive"),
    ("FDX",  "AMZN",  "logistics",               "logistics"),
    ("UPS",  "WMT",   "logistics",               "logistics"),
    ("UPS",  "TGT",   "logistics",               "logistics"),
    ("FDX",  "TGT",   "logistics",               "logistics"),
    ("ZIM",  "WMT",   "ocean_freight",           "logistics"),
    ("INTC", "NVDA",  "chip_competitor_partner", "semiconductors"),
]

COMPANY_METADATA = {
    "TSM":  {"name": "Taiwan Semiconductor", "sector": "semiconductors", "tier": 1, "country": "TW"},
    "NVDA": {"name": "NVIDIA",               "sector": "semiconductors", "tier": 2, "country": "US"},
    "ASML": {"name": "ASML",                 "sector": "semiconductors", "tier": 0, "country": "NL"},
    "AMAT": {"name": "Applied Materials",    "sector": "semiconductors", "tier": 0, "country": "US"},
    "MU":   {"name": "Micron Technology",    "sector": "semiconductors", "tier": 1, "country": "US"},
    "INTC": {"name": "Intel",                "sector": "semiconductors", "tier": 1, "country": "US"},
    "HON":  {"name": "Honeywell",            "sector": "manufacturing",  "tier": 1, "country": "US"},
    "GE":   {"name": "GE",                   "sector": "manufacturing",  "tier": 2, "country": "US"},
    "EMR":  {"name": "Emerson Electric",     "sector": "manufacturing",  "tier": 0, "country": "US"},
    "MMM":  {"name": "3M",                   "sector": "manufacturing",  "tier": 1, "country": "US"},
    "F":    {"name": "Ford",                 "sector": "automotive",     "tier": 2, "country": "US"},
    "GM":   {"name": "General Motors",       "sector": "automotive",     "tier": 2, "country": "US"},
    "TM":   {"name": "Toyota",               "sector": "automotive",     "tier": 2, "country": "JP"},
    "AA":   {"name": "Alcoa",                "sector": "raw_materials",  "tier": 0, "country": "US"},
    "FCX":  {"name": "Freeport-McMoRan",     "sector": "raw_materials",  "tier": 0, "country": "US"},
    "NUE":  {"name": "Nucor",                "sector": "raw_materials",  "tier": 0, "country": "US"},
    "CF":   {"name": "CF Industries",        "sector": "raw_materials",  "tier": 0, "country": "US"},
    "FDX":  {"name": "FedEx",                "sector": "logistics",      "tier": 1, "country": "US"},
    "UPS":  {"name": "UPS",                  "sector": "logistics",      "tier": 1, "country": "US"},
    "ZIM":  {"name": "ZIM Shipping",         "sector": "logistics",      "tier": 1, "country": "IL"},
    "WMT":  {"name": "Walmart",              "sector": "retail",         "tier": 3, "country": "US"},
    "TGT":  {"name": "Target",               "sector": "retail",         "tier": 3, "country": "US"},
    "AMZN": {"name": "Amazon",               "sector": "retail",         "tier": 3, "country": "US"},
    "COST": {"name": "Costco",               "sector": "retail",         "tier": 3, "country": "US"},
    "AAPL": {"name": "Apple",                "sector": "semiconductors", "tier": 3, "country": "US"},
}

def build_graph(stress_scores: dict = None) -> nx.DiGraph:
    G = nx.DiGraph()
    for ticker, meta in COMPANY_METADATA.items():
        G.add_node(ticker, **meta,
                   stress_score=stress_scores.get(ticker, 0.0) if stress_scores else 0.0)
    for supplier, customer, rel_type, sector in SUPPLY_CHAIN_EDGES:
        G.add_edge(supplier, customer, relationship=rel_type, sector=sector)
    return G

def propagate_stress(G: nx.DiGraph, initial_stress: dict,
                     damping: float = 0.6, iterations: int = 3) -> dict:
    stress = {n: initial_stress.get(n, 0.0) for n in G.nodes()}
    for _ in range(iterations):
        new_stress = dict(stress)
        for node in G.nodes():
            upstream = sum(stress[p] * damping for p in G.predecessors(node))
            new_stress[node] = min(1.0, stress[node] + upstream / max(1, G.in_degree(node)))
        stress = new_stress
    nx.set_node_attributes(G, stress, "propagated_stress")
    return stress

def compute_centrality_features(G: nx.DiGraph) -> pd.DataFrame:
    betweenness = nx.betweenness_centrality(G)
    pagerank    = nx.pagerank(G)
    in_deg      = dict(G.in_degree())
    out_deg     = dict(G.out_degree())
    return pd.DataFrame([{
        "ticker":      n,
        "betweenness": betweenness.get(n, 0),
        "pagerank":    pagerank.get(n, 0),
        "in_degree":   in_deg.get(n, 0),
        "out_degree":  out_deg.get(n, 0),
        "tier":        G.nodes[n].get("tier", 0),
    } for n in G.nodes()])