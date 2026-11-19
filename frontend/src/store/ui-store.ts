import { create } from "zustand";
import i18n from "@/i18n";

type Theme = "light" | "dark";
type Locale = "ru" | "en";

type UiState = {
  theme: Theme;
  locale: Locale;
  backendOnline: boolean;
  lastUpdated: string | null;
  sidebarCollapsed: boolean;
  setTheme: (theme: Theme) => void;
  setLocale: (locale: Locale) => void;
  setConnection: (online: boolean, updatedAt: string | null) => void;
  toggleSidebar: () => void;
};

export const useUiStore = create<UiState>((set) => ({
  theme: localStorage.getItem("abylkhaiyr-theme") === "light" ? "light" : "dark",
  locale: i18n.language === "en" ? "en" : "ru",
  backendOnline: true,
  lastUpdated: null,
  sidebarCollapsed: localStorage.getItem("abylkhaiyr-sidebar") === "collapsed",
  setTheme: (theme) => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("abylkhaiyr-theme", theme);
    set({ theme });
  },
  setLocale: (locale) => {
    localStorage.setItem("abylkhaiyr-language", locale);
    void i18n.changeLanguage(locale);
    set({ locale });
  },
  setConnection: (backendOnline, lastUpdated) => set({ backendOnline, lastUpdated }),
  toggleSidebar: () => set((state) => {
    const sidebarCollapsed = !state.sidebarCollapsed;
    localStorage.setItem("abylkhaiyr-sidebar", sidebarCollapsed ? "collapsed" : "expanded");
    return { sidebarCollapsed };
  }),
}));
