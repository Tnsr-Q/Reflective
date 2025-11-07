import React from "react";

export default function TrustMeter({ trust }) {
  const personas = ["honest", "bluffer", "chaotic"];
  return (
    <div className="panel" data-testid="trust-meter">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3>Opponent Trust</h3>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
        {personas.map((p) => {
          const t = trust?.[p] || { accuracy: 0, correct: 0, total: 0 };
          const pct = Math.round((t.accuracy || 0) * 100);
          return (
            <div key={p} className="panel" style={{ padding: 12 }} data-testid={`trust-card-${p}`}>
              <div style={{ fontWeight: 600, textTransform: "capitalize" }}>{p}</div>
              <div style={{ height: 6 }} />
              <div style={{ background: "#0b1220", border: "1px solid #334155", borderRadius: 6, height: 8 }}>
                <div style={{ width: `${pct}%`, height: 8, background: "#22c55e", borderRadius: 6 }} />
              </div>
              <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 6 }}>{t.correct}/{t.total} correct</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
