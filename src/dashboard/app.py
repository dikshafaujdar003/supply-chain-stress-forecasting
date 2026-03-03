import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Windows-safe path resolution
sys.path.append(str(Path(__file__).resolve().parents[2]))

from streamlit_agraph import agraph, Node, Edge, Config
from src.rag.qa_chain import build_qa_chain, ask
from src.processing.graph_builder import build_graph, propagate_stress, COMPANY_METADATA

st.set_page_config(
    page_title="Supply Chain Stress Monitor",
    layout="wide",
    page_icon="🏭"
)

@st.cache_data
def load_data():
    return (
        pd.read_parquet(Path("data") / "processed" / "current_stress.parquet"),
        pd.read_parquet(Path("data") / "processed" / "forecast.parquet"),
        pd.read_parquet(Path("data") / "processed" / "daily_sentiment.parquet"),
    )

@st.cache_resource
def load_qa():
    return build_qa_chain()

stress_df, forecast_df, sentiment_df = load_data()

st.sidebar.title("🏭 Supply Chain Stress Monitor")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🕸️ Network Graph",
    "🔮 Forecast",
    "💬 Ask the Assistant"
])

# ── Dashboard ──────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("Supply Chain Stress Monitor")
    st.caption("Real-time stress signals across supplier networks")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 High Stress",
              int((stress_df["stress_probability"] > 0.7).sum()))
    c2.metric("🟡 Medium Stress",
              int(((stress_df["stress_probability"] > 0.4) &
                   (stress_df["stress_probability"] <= 0.7)).sum()))
    c3.metric("📈 Avg Stress",
              f"{stress_df['stress_probability'].mean():.1%}")
    c4.metric("🏭 Companies", len(stress_df))

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Stress by Company")
        fig = px.bar(
            stress_df.sort_values("stress_probability"),
            x="stress_probability",
            y="ticker",
            color="stress_probability",
            color_continuous_scale=["green", "yellow", "red"],
            range_color=[0, 1],
            orientation="h",
            height=500,
            labels={"stress_probability": "Stress Score", "ticker": ""},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Stress by Sector")
        sector_stress = (stress_df.groupby("sector")["stress_probability"]
                         .mean().reset_index())
        fig2 = px.pie(
            sector_stress,
            values="stress_probability",
            names="sector",
            color_discrete_sequence=px.colors.sequential.RdBu_r,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("News Sentiment Trend")
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sentiment_df["date"],
        y=sentiment_df["avg_sentiment"],
        mode="lines",
        name="Avg Sentiment",
        line=dict(color="blue"),
    ))
    fig3.add_trace(go.Scatter(
        x=sentiment_df["date"],
        y=-sentiment_df["negative_ratio"],
        mode="lines",
        name="Negative Ratio",
        line=dict(color="red", dash="dot"),
    ))
    fig3.add_hline(y=0, line_dash="dash", line_color="gray")
    fig3.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)

# ── Network Graph ──────────────────────────────────────────────────────────────
elif page == "🕸️ Network Graph":
    st.title("🕸️ Supply Chain Network")
    st.caption("Node color = stress (red=high). Arrows = supplier → customer.")

    stress_dict = dict(zip(
        stress_df["ticker"],
        stress_df.get("propagated_stress", stress_df["stress_probability"])
    ))
    G = build_graph(stress_dict)

    def node_color(s):
        return "#ef4444" if s > 0.7 else "#f59e0b" if s > 0.4 else "#22c55e"

    nodes = [
        Node(
            id=n,
            label=f"{n}\n{stress_dict.get(n, 0):.0%}",
            size=20 + G.degree(n) * 5,
            color=node_color(stress_dict.get(n, 0)),
            title=COMPANY_METADATA.get(n, {}).get("name", n),
        )
        for n in G.nodes()
    ]
    edges = [
        Edge(source=u, target=v, label=G[u][v].get("relationship", ""))
        for u, v in G.edges()
    ]
    agraph(
        nodes=nodes,
        edges=edges,
        config=Config(width="100%", height=600, directed=True, physics=True),
    )

# ── Forecast ───────────────────────────────────────────────────────────────────
elif page == "🔮 Forecast":
    st.title("🔮 30-Day Stress Forecast")

    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(0,100,255,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Interval",
    ))
    # Fix: use timestamp in milliseconds for plotly vline
    today_ms = pd.Timestamp.today().timestamp() * 1000
    fig.add_vline(
        x=today_ms,
        line_dash="dash",
        annotation_text="Today",
        line_color="red",
    )
    fig.update_layout(
        height=450,
        yaxis_title="Stress Index",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30),
        use_container_width=True,
    )

# ── RAG Assistant ──────────────────────────────────────────────────────────────
elif page == "💬 Ask the Assistant":
    st.title("💬 Supply Chain Q&A Assistant")
    st.caption("Powered by Llama 3.1 70B via Groq + local embeddings.")

    EXAMPLES = [
        "Which companies are at highest risk right now and why?",
        "What are the main drivers of semiconductor supply chain stress?",
        "How is the logistics sector being affected?",
        "What did recent SEC filings say about automotive supply chain risk?",
        "What is the 30-day stress forecast and what's driving it?",
    ]
    st.markdown("**Try one of these:**")
    for q in EXAMPLES:
        if st.button(q, key=q):
            st.session_state["q"] = q

    question = st.text_input(
        "Your question:",
        value=st.session_state.get("q", ""),
        placeholder="e.g. Which semiconductor companies face highest supply chain risk?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Searching documents and generating answer..."):
            qa = load_qa()
            result = ask(question, qa)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.subheader("Sources")
            seen = set()
            for src in result["sources"]:
                key = src.get("url") or src.get("source")
                if key in seen:
                    continue
                seen.add(key)
                if src.get("url"):
                    st.markdown(
                        f"- [{src['source']}]({src['url']}) "
                        f"— {str(src.get('date',''))[:10]}"
                    )
                else:
                    st.markdown(f"- {src['source']} ({src['type']})")

        st.caption("LLM: Llama 3.3 70B via Groq | Embeddings: all-MiniLM-L6-v2 (local)")