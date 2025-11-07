import React from "react";

export default function Ticker({ price, change24h, nextPredictionAt }) {
  const changeColor = change24h > 0 ? "#22c55e" : change24h < 0 ? "#ef4444" : "#94a3b8";
  const fmtPrice = price ? `$${Number(price).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "—";
  const fmtChange = change24h != null ? `${change24h >= 0 ? "+" : ""}${change24h.toFixed(2)}%` : "—";

  const [countdown, setCountdown] = React.useState("—");
  React.useEffect(() => {
    if (!nextPredictionAt) return;
    const t = setInterval(() => {
      const now = Date.now();
      const target = new Date(nextPredictionAt).getTime();
      const diff = Math.max(0, Math.floor((target - now) / 1000));
      const h = String(Math.floor(diff / 3600)).padStart(2, "0");
      const m = String(Math.floor((diff % 3600) / 60)).padStart(2, "0");
      const s = String(diff % 60).padStart(2, "0");
      setCountdown(`${h}:${m}:${s}`);
    }, 1000);
    return () => clearInterval(t);
  }, [nextPredictionAt]);

  return (
    <div className="panel" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <div data-testid="ticker-price">
        <strong>BTC</strong> {fmtPrice} <span style={{ color: changeColor }}>({fmtChange})</span>
      </div>
      <div data-testid="next-prediction">Next prediction in {countdown}</div>
    </div>
  );
}
