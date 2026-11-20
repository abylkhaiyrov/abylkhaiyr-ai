import { createContext, useContext } from "react";
import type { GenerationEvent } from "@/lib/events";

export type EventClient = {
  send: (event: Record<string, unknown>) => boolean;
  subscribe: (listener: (event: GenerationEvent) => void) => () => void;
};

export const EventContext = createContext<EventClient | null>(null);

export function useStudioEvents() {
  const value = useContext(EventContext);
  if (!value) throw new Error("StudioEventsProvider is missing");
  return value;
}
