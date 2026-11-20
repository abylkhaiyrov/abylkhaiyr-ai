import { useEffect, useMemo, useRef, useState } from "react";
import { Bell, CheckCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { NotificationList } from "@/components/notifications/notification-list";
import { Button } from "@/components/ui/button";
import { useNotificationActions, useNotificationList, useUnreadCount } from "@/hooks/use-notifications";

const PREVIEW = 8;

export function NotificationBell() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const unread = useUnreadCount().data ?? 0;
  const query = useMemo(() => ({ unread: false, type: null, limit: PREVIEW, offset: 0 }), []);
  const list = useNotificationList(query, open);
  const actions = useNotificationActions();
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
  return (
    <div ref={ref} className="relative">
      <Button variant="ghost" size="icon" aria-haspopup="dialog" aria-expanded={open} aria-label={unread > 0 ? t("notification.bellUnread", { count: unread }) : t("notification.title")} onClick={() => setOpen((value) => !value)}>
        <Bell size={16} />
        {unread > 0 ? <span className="absolute right-0.5 top-0.5 grid min-h-4 min-w-4 place-items-center rounded-full bg-accent px-1 text-[10px] font-semibold leading-none text-accent-foreground">{unread > 99 ? "99+" : unread}</span> : null}
      </Button>
      {open ? (
        <div role="dialog" aria-label={t("notification.title")} className="absolute right-0 top-full z-40 mt-1 w-[22rem] max-w-[92vw] rounded-xl border bg-background shadow-xl animate-rise">
          <div className="flex items-center justify-between border-b hairline px-3 py-2">
            <p className="text-[13px] font-semibold">{t("notification.title")}</p>
            <Button size="sm" variant="ghost" disabled={unread === 0 || actions.readAll.isPending} onClick={() => actions.readAll.mutate()}><CheckCheck size={13} />{t("notification.markAll")}</Button>
          </div>
          <div className="max-h-[22rem] overflow-y-auto"><NotificationList query={list} compact onNavigate={() => setOpen(false)} /></div>
          <div className="border-t hairline p-2 text-center">
            <Link to="/notifications" onClick={() => setOpen(false)} className="text-xs font-medium text-accent hover:underline">{t("notification.viewAll")}</Link>
          </div>
        </div>
      ) : null}
    </div>
  );
}
