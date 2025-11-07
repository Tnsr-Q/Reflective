import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function Graph({ data, onNodeClick }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!data || !data.nodes || !data.links) return;

    const width = 1100;
    const height = 600;

    const svg = d3
      .select(ref.current)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", width)
      .attr("height", height)
      .attr("data-testid", "force-graph-svg");

    svg.selectAll("*").remove();

    const color = (d) => {
      if (d.type === "project") return "#38bdf8"; // cyan
      if (d.type === "reflection") return "#22c55e"; // green
      if (d.type === "anomaly") {
        switch (d.severity) {
          case "critical":
            return "#ef4444";
          case "high":
            return "#f97316";
          case "medium":
            return "#eab308";
          default:
            return "#a3a3a3";
        }
      }
      return "#94a3b8";
    };

    const simulation = d3
      .forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id((d) => d.id).distance(90))
      .force("charge", d3.forceManyBody().strength(-160))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append("g")
      .attr("stroke", "#475569")
      .attr("stroke-opacity", 0.4)
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("data-testid", (d) => `graph-link-${d.source}-${d.target}`)
      .attr("stroke-width", 1.4);

    const node = svg
      .append("g")
      .attr("stroke", "#0ea5e9")
      .attr("stroke-width", 1)
      .selectAll("circle")
      .data(data.nodes)
      .join("circle")
      .attr("r", 8)
      .attr("fill", color)
      .attr("data-testid", (d) => `graph-node-${d.id}`)
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      )
      .on("click", (_, d) => onNodeClick && onNodeClick(d));

    const labels = svg
      .append("g")
      .selectAll("text")
      .data(data.nodes)
      .join("text")
      .text((d) => d.label || d.id)
      .attr("font-size", 10)
      .attr("fill", "#e2e8f0");

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      labels.attr("x", (d) => d.x + 10).attr("y", (d) => d.y + 4);
    });

    return () => simulation.stop();
  }, [data, onNodeClick]);

  return <svg ref={ref} />;
}
