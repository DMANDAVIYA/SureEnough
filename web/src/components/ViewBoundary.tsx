import { Component, type ReactNode } from 'react';

interface State {
  error: Error | null;
}

export default class ViewBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <p className="status" role="alert">
        This view could not be shown ({error.message}). The exported data does not match what the page reads.
      </p>
    );
  }
}
