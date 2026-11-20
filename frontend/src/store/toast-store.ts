import { create } from "zustand";

export type ToastAction = { label: string; href: string };
export type Toast = { id: number; kind: "success" | "error" | "info"; message: string; action?: ToastAction };

type ToastState = {
  toasts: Toast[];
  push: (kind: Toast["kind"], message: string, action?: ToastAction) => void;
  dismiss: (id: number) => void;
};

let counter = 0;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (kind, message, action) => {
    counter += 1;
    const id = counter;
    set((state) => ({ toasts: [...state.toasts.slice(-3), { id, kind, message, action }] }));
    setTimeout(() => set((state) => ({ toasts: state.toasts.filter((item) => item.id !== id) })), 5000);
  },
  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((item) => item.id !== id) })),
}));
