import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, message: String(error?.message || error) };
  }
  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error("UI error boundary:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="panel" style={{ margin: 24 }} data-testid="global-error-boundary">
          <h3>Something went wrong</h3>
          <div style={{ fontSize: 12, color: "#94a3b8" }}>{this.state.message}</div>
        </div>
      );
    }
    // eslint-disable-next-line react/prop-types
    return this.props.children;
  }
}
