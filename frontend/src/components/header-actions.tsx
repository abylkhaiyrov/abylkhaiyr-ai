import { useEffect, useRef, useState } from "react";
import { LogOut, Moon, Settings, Sun, UserRound } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { NotificationBell } from "@/components/notifications/notification-bell";
import { Button } from "@/components/ui/button";
import { useCurrentUser } from "@/hooks/use-auth";
import { api } from "@/lib/api";
import { Select } from "@/components/ui/select";
import { useUiStore } from "@/store/ui-store";

function UserMenu() {
  const { t } = useTranslation();
  const user = useCurrentUser();
  const navigate = useNavigate();
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const logout = useMutation({
    mutationFn: api.logout,
    onSettled: () => {
      client.clear();
      navigate("/login");
    },
  });
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => !ref.current?.contains(event.target as Node) && setOpen(false);
    const key = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  const initials = `${user.first_name[0] ?? ""}${user.last_name[0] ?? ""}`.toUpperCase() || user.username[0].toUpperCase();
  const item = "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] hover:bg-muted";
  return (
    <div ref={ref} className="relative">
      <button type="button" aria-haspopup="menu" aria-expanded={open} aria-label={t("account.menu")} onClick={() => setOpen((value) => !value)} className="ml-1 grid h-8 w-8 place-items-center rounded-full bg-muted text-xs font-semibold hover:bg-muted/70">{initials}</button>
      {open ? (
        <div role="menu" className="absolute right-0 top-full z-40 mt-1 w-56 rounded-xl border bg-background p-1 shadow-xl animate-rise" onClick={() => setOpen(false)}>
          <div className="px-2.5 py-2">
            <p className="truncate text-[13px] font-semibold">{`${user.first_name} ${user.last_name}`.trim() || user.username}</p>
            <p className="truncate text-xs text-muted-foreground">{user.email ?? user.username}</p>
            <p className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">{t(`role.account.${user.role}`)}</p>
          </div>
          <Link role="menuitem" className={item} to="/profile"><UserRound size={14} />{t("account.profile")}</Link>
          <Link role="menuitem" className={item} to="/settings"><Settings size={14} />{t("nav.settings")}</Link>
          <button role="menuitem" type="button" className={item} onClick={() => logout.mutate()}><LogOut size={14} />{t("app.logout")}</button>
        </div>
      ) : null}
    </div>
  );
}

export function HeaderActions() {
  const { t } = useTranslation();
  const { theme, locale, setTheme, setLocale } = useUiStore();
  return (
    <div className="flex items-center gap-0.5">
      <Select compact aria-label={t("settings.language")} value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")}>
        <option value="ru">RU</option>
        <option value="en">EN</option>
      </Select>
      <Button variant="ghost" size="icon" aria-label={t("settings.theme")} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</Button>
      <NotificationBell />
      <UserMenu />
    </div>
  );
}
