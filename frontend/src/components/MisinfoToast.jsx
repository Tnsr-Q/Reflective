import React from "react";

export default function MisinfoToast({ item, onClose }) {
  if (!item) return null;
  return (
    <div className="panel" data-testid="misinfo-toast" style={{ position: "fixed", top: 16, right: 16, zIndex: 1000, borderColor: "#ef4444" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span style={{ fontSize: 18 }}>⚠️</span>
        <div>
          <div style={{ fontWeight: 600 }}>Unverified tweet injected</div>
          <div style={{ fontSize: 12, color: "#94a3b8", maxWidth: 320 }}>{item.headline}</div>
        </div>
        <button data-testid="misinfo-toast-close" className="button" onClick={onClose}>Dismiss</button>
      </div>
    </div>
  );
}
