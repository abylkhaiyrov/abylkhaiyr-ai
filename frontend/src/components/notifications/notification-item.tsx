import { AlertTriangle, Ban, CheckCircle2, Info, Trash2, UserRound } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { useFormat } from "@/hooks/use-format";
import type { AppNotification, NotificationType } from "@/lib/api";
import { cn } from "@/lib/utils";

const ICONS: Record<NotificationType, { icon: typeof Info; tone: string }> = {
  GENERATION_COMPLETED: { icon: CheckCircle2, tone: "text-success" },
  GENERATION_FAILED: { icon: AlertTriangle, tone: "text-destructive" },
  GENERATION_CANCELLED: { icon: Ban, tone: "text-muted-foreground" },
  SYSTEM: { icon: Info, tone: "text-info" },
  ACCOUNT: { icon: UserRound, tone: "text-accent" },
};

type Props = {
  notification: AppNotification;
  onOpen: (notification: AppNotification) => void;
  onDelete: (notification: AppNotification) => void;
  compact?: boolean;
};

export function NotificationItem({ notification, onOpen, onDelete, compact = false }: Props) {
  const { t, i18n } = useTranslation();
  const format = useFormat();
  const { icon: Icon, tone } = ICONS[notification.type];
  const generation = notification.type.startsWith("GENERATION");
  const titleKey = generation ? `notification.${notification.type}` : `notification.codes.${notification.title}`;
  const textKey = generation ? `notification.${notification.type}Text` : `notification.codes.${notification.title}Text`;
  const title = i18n.exists(titleKey) ? t(titleKey) : notification.title;
  const description = i18n.exists(textKey) ? t(textKey) : notification.message;
  const subtitle = generation ? notification.message : "";
  return (
    <li className={cn("group relative flex items-start gap-3 rounded-lg px-3 py-2.5 transition-colors hover:bg-muted/60", !notification.is_read && "bg-accent/5")}>
      <Icon size={16} className={cn("mt-0.5 shrink-0", tone)} aria-hidden />
      <button type="button" className="min-w-0 flex-1 text-left" onClick={() => onOpen(notification)}>
        <span className="flex items-center gap-2">
          <span className={cn("truncate text-[13px]", !notification.is_read ? "font-semibold" : "font-medium")}>{title}</span>
          {!notification.is_read ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-label={t("notification.unread")} /> : null}
        </span>
        <span className="block truncate text-xs text-muted-foreground">{description}</span>
        {subtitle && !compact ? <span className="block truncate text-xs text-muted-foreground/80">{subtitle}</span> : null}
        <span className="mt-0.5 block text-[11px] text-muted-foreground">{format.relative(notification.created_at)}</span>
      </button>
      <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 transition-opacity focus-visible:opacity-100 group-hover:opacity-100" aria-label={t("notification.delete")} onClick={() => onDelete(notification)}>
        <Trash2 size={13} />
      </Button>
    </li>
  );
}
