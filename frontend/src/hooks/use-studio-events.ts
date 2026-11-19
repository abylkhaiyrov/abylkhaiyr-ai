import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { useUiStore } from "@/store/ui-store";

const liveKeys = ["dashboard", "system", "processes", "metrics", "logs"];

export function useStudioEvents() {
  const client = useQueryClient();
  const setConnection = useUiStore((state) => state.setConnection);
  const lastRefresh = useRef(0);

  useEffect(() => {
    const source = new EventSource("/api/events");
    const received = () => {
      setConnection(true, new Date().toISOString());
      const now = Date.now();
      if (now - lastRefresh.current < 10_000) return;
      lastRefresh.current = now;
      for (const key of liveKeys) {
        void client.invalidateQueries({ queryKey: [key], refetchType: "active" });
      }
    };
    const disconnected = () => setConnection(false, null);
    source.addEventListener("studio", received);
    source.addEventListener("error", disconnected);
    return () => source.close();
  }, [client, setConnection]);
}
