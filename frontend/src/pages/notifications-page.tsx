import { useMemo, useState } from "react";
import { CheckCheck } from "lucide-react";
import { useTranslation } from "react-i18next";
import { PageHeader } from "@/components/data-state";
import { NotificationList } from "@/components/notifications/notification-list";
import { Button } from "@/components/ui/button";
import { Tabs } from "@/components/ui/tabs";
import { useNotificationActions, useNotificationList } from "@/hooks/use-notifications";
import { Select } from "@/components/ui/select";
import type { NotificationType } from "@/lib/api";

const PAGE = 20;
const TYPES: NotificationType[] = ["GENERATION_COMPLETED", "GENERATION_FAILED", "GENERATION_CANCELLED", "SYSTEM", "ACCOUNT"];

export function NotificationsPage() {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<"all" | "unread">("all");
  const [type, setType] = useState<NotificationType | "">("");
  const [page, setPage] = useState(0);
  const query = useMemo(() => ({ unread: filter === "unread", type: type || null, limit: PAGE, offset: page * PAGE }), [filter, type, page]);
  const list = useNotificationList(query);
  const actions = useNotificationActions();
  const total = list.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  return (
    <div className="grid gap-5">
      <PageHeader
        title={t("notification.title")}
        subtitle={t("notification.subtitle", { unread: list.data?.unread ?? 0 })}
        actions={<Button size="sm" variant="outline" disabled={(list.data?.unread ?? 0) === 0 || actions.readAll.isPending} onClick={() => actions.readAll.mutate()}><CheckCheck size={14} />{t("notification.markAll")}</Button>}
      />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Tabs value={filter} onChange={(value) => { setFilter(value); setPage(0); }} items={[{ value: "all", label: t("notification.filterAll") }, { value: "unread", label: t("notification.filterUnread") }]} />
        <Select inline aria-label={t("notification.type")} value={type} onChange={(event) => { setType(event.target.value as NotificationType | ""); setPage(0); }}>
          <option value="">{t("notification.allTypes")}</option>
          {TYPES.map((item) => <option key={item} value={item}>{t(`notification.types.${item}`)}</option>)}
        </Select>
      </div>
      <NotificationList query={list} />
      {pages > 1 ? (
        <div className="flex items-center justify-between text-[13px] text-muted-foreground">
          <span>{t("common.pageOf", { page: page + 1, pages })}</span>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" disabled={page === 0} onClick={() => setPage(page - 1)}>{t("common.previous")}</Button>
            <Button size="sm" variant="outline" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>{t("common.next")}</Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
