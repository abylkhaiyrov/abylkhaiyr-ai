import { create } from "zustand";
import i18n from "@/i18n";

export type SocketStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

type Theme = "light" | "dark";
type Locale = "ru" | "en";

type UiState = {
  theme: Theme;
  locale: Locale;
  socketStatus: SocketStatus;
  sidebarCollapsed: boolean;
  historyOpen: boolean;
  toggleHistory: () => void;
  setTheme: (theme: Theme) => void;
  setLocale: (locale: Locale) => void;
  setSocketStatus: (status: SocketStatus) => void;
  toggleSidebar: () => void;
};

const read = (key: string) => {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};
const write = (key: string, value: string) => {
  try {
    localStorage.setItem(key, value);
  } catch {
    return;
  }
};

export const useUiStore = create<UiState>((set) => ({
  theme: read("abylkhaiyr-theme") === "light" ? "light" : "dark",
  locale: i18n.language === "en" ? "en" : "ru",
  socketStatus: "connecting",
  sidebarCollapsed: read("abylkhaiyr-sidebar") === "collapsed",
  historyOpen: read("abylkhaiyr-history") !== "closed",
  toggleHistory: () =>
    set((state) => {
      write("abylkhaiyr-history", state.historyOpen ? "closed" : "open");
      return { historyOpen: !state.historyOpen };
    }),
  setTheme: (theme) => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    write("abylkhaiyr-theme", theme);
    set({ theme });
  },
  setLocale: (locale) => {
    write("abylkhaiyr-language", locale);
    document.documentElement.lang = locale;
    void i18n.changeLanguage(locale);
    set({ locale });
  },
  setSocketStatus: (socketStatus) => set((state) => (state.socketStatus === socketStatus ? state : { socketStatus })),
  toggleSidebar: () =>
    set((state) => {
      const sidebarCollapsed = !state.sidebarCollapsed;
      write("abylkhaiyr-sidebar", sidebarCollapsed ? "collapsed" : "expanded");
      return { sidebarCollapsed };
    }),
}));
