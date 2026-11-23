import type { Config } from "tailwindcss";

const token = (name: string) => `hsl(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: ["class"],
  content: {
    relative: true,
    files: ["./index.html", "./src/**/*.{ts,tsx}"],
  },
  theme: {
    extend: {
      colors: {
        background: token("background"),
        foreground: token("foreground"),
        card: token("card"),
        "card-foreground": token("card-foreground"),
        muted: token("muted"),
        "muted-foreground": token("muted-foreground"),
        border: token("border"),
        primary: token("primary"),
        "primary-foreground": token("primary-foreground"),
        destructive: token("destructive"),
        accent: token("accent"),
        "accent-foreground": token("accent-foreground"),
        success: token("success"),
        warning: token("warning"),
        info: token("info"),
      },
      fontFamily: {
        sans: ["Inter Variable", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Ubuntu", "Cantarell", "Noto Sans", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "monospace"],
      },
      borderRadius: { md: "0.5rem", lg: "0.625rem", xl: "0.875rem", "2xl": "1.25rem" },
      keyframes: {
        indeterminate: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(350%)" },
        },
        caret: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0" } },
        rise: { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
        fade: { from: { opacity: "0" }, to: { opacity: "1" } },
      },
      animation: {
        indeterminate: "indeterminate 1.4s ease-in-out infinite",
        caret: "caret 1s steps(1) infinite",
        rise: "rise 160ms ease-out",
        fade: "fade 220ms ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
