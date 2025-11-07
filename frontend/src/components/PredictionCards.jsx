import React from "react";

const dirIcon = (d) => (d === 2 ? "↑" : d === 0 ? "↓" : "→");
const sentColor = (s) => (s === "bullish" ? "#22c55e" : s === "bearish" ? "#ef4444" : "#94a3b8");

export default function PredictionCards({ items }) {
  const horizonsOrder = ["1h", "4h", "8h", "24h", "3d", "2w", "1m"];
  const sorted = [...(items || [])].sort((a, b) => horizonsOrder.indexOf(a.horizon) - horizonsOrder.indexOf(b.horizon));

  return (
    <div className="grid" style={{ gridTemplateColumns: "repeat(7, minmax(0, 1fr))", gap: 12 }}>
      {sorted.map((p) => (
        <div key={`${p.horizon}-${p.target_ts || "now"}`} className="panel" data-testid={`prediction-card-${p.horizon}`}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ fontWeight: 600 }}>{p.horizon}</div>
            <div style={{ fontSize: 18 }}>{dirIcon(p.direction)}</div>
          </div>
          <div style={{ height: 8 }} />
          <div>
            <div style={{ fontSize: 12, color: "#94a3b8" }}>confidence</div>
            <div style={{ background: "#0b1220", border: "1px solid #334155", borderRadius: 6, height: 8 }}>
              <div style={{ width: `${Math.round((p.confidence || 0) * 100)}%`, height: 8, background: "#06b6d4", borderRadius: 6 }} />
            </div>
          </div>
          <div style={{ height: 8 }} />
          <div>
            <span className="badge" style={{ background: sentColor(p.sentiment), color: "#0b1220" }} data-testid={`prediction-sent-${p.horizon}`}>
              {p.sentiment || "neutral"}
            </span>
          </div>
          <div style={{ height: 8 }} />
          <div style={{ fontSize: 12, color: "#94a3b8" }} title={p.reasoning_text}>
            {p.reasoning_text ? `${p.reasoning_text.substring(0, 80)}${p.reasoning_text.length > 80 ? "…" : ""}` : ""}
          </div>
        </div>
      ))}
    </div>
  );
}
