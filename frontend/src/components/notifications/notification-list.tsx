import { BellOff } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/data-state";
import { NotificationItem } from "@/components/notifications/notification-item";
import { useNotificationActions } from "@/hooks/use-notifications";
import type { AppNotification, NotificationPage } from "@/lib/api";

type Props = {
  query: { data: NotificationPage | undefined; isPending: boolean; isError: boolean; error: Error | null; refetch: () => unknown };
  compact?: boolean;
  onNavigate?: () => void;
};

export function NotificationList({ query, compact = false, onNavigate }: Props) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const actions = useNotificationActions();
  const open = (notification: AppNotification) => {
    if (!notification.is_read) actions.read.mutate(notification.id);
    if (notification.chat_id) {
      navigate(`/chat?c=${notification.chat_id}`);
      onNavigate?.();
    }
  };
  if (query.isPending) {
    return (
      <div role="status" aria-busy="true" aria-label={t("common.loading")} className="grid gap-2 p-3">
        {[0, 1, 2].map((item) => <div key={item} className="h-12 animate-pulse rounded-lg bg-muted" />)}
      </div>
    );
  }
  if (query.isError || !query.data) return <div className="p-3"><ErrorState error={query.error ?? new Error("unknown")} retry={() => void query.refetch()} /></div>;
  if (query.data.items.length === 0) {
    return (
      <div className="grid place-items-center gap-1 px-4 py-10 text-center">
        <BellOff size={20} className="text-muted-foreground" />
        <p className="text-[13px] font-medium">{t("notification.empty")}</p>
        <p className="text-xs text-muted-foreground">{t("notification.emptyHint")}</p>
      </div>
    );
  }
  return (
    <ul className="grid gap-0.5 p-1">
      {query.data.items.map((item) => (
        <NotificationItem key={item.id} notification={item} compact={compact} onOpen={open} onDelete={(target) => actions.remove.mutate(target.id)} />
      ))}
    </ul>
  );
}
