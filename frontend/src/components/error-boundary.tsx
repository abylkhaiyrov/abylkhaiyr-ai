import { Component, type ErrorInfo, type ReactNode } from "react";
import i18n from "@/i18n";
import { Button } from "@/components/ui/button";

type Props = { children: ReactNode };
type State = { failed: boolean };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Page rendering failed", error, info.componentStack);
  }

  render() {
    if (this.state.failed) {
      return (
        <main role="alert" className="mx-auto max-w-xl p-6">
          <p>{i18n.t("errors.render")}</p>
          <Button className="mt-4" onClick={() => window.location.reload()}>{i18n.t("errors.reload")}</Button>
        </main>
      );
    }

    return this.props.children;
  }
}
