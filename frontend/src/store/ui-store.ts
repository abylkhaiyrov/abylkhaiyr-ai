import { create } from "zustand";
import i18n from "@/i18n";

type Theme = "light" | "dark";
type Locale = "ru" | "en";

type UiState = {
  theme: Theme;
  locale: Locale;
  backendOnline: boolean;
  sidebarCollapsed: boolean;
  setTheme: (theme: Theme) => void;
  setLocale: (locale: Locale) => void;
  setConnection: (online: boolean) => void;
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
  backendOnline: true,
  sidebarCollapsed: read("abylkhaiyr-sidebar") === "collapsed",
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
  setConnection: (backendOnline) => set((state) => (state.backendOnline === backendOnline ? state : { backendOnline })),
  toggleSidebar: () =>
    set((state) => {
      const sidebarCollapsed = !state.sidebarCollapsed;
      write("abylkhaiyr-sidebar", sidebarCollapsed ? "collapsed" : "expanded");
      return { sidebarCollapsed };
    }),
}));
