import { Component, type ErrorInfo, type ReactNode } from "react";
import i18n from "@/i18n";
import { Button } from "@/components/ui/button";

type Props = { children: ReactNode; page?: boolean };
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
        <section role="alert" className="mx-auto max-w-xl p-6">
          <p>{i18n.t("errors.render")}</p>
          {this.props.page ? (
            <Button className="mt-4 mr-2" variant="outline" onClick={() => this.setState({ failed: false })}>{i18n.t("common.retry")}</Button>
          ) : null}
          <Button className="mt-4" onClick={() => window.location.reload()}>{i18n.t("errors.reload")}</Button>
        </section>
      );
    }

    return this.props.children;
  }
}
