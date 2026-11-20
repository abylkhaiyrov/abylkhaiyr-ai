import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useApiError } from "@/hooks/use-api-error";
import { api, type NotificationQuery } from "@/lib/api";
import { useToastStore } from "@/store/toast-store";

export const useUnreadCount = () =>
  useQuery({ queryKey: ["notifications", "count"], queryFn: api.unreadCount, staleTime: Infinity, select: (data) => data.unread });

export const useNotificationList = (query: NotificationQuery, enabled = true) =>
  useQuery({ queryKey: ["notifications", "list", query], queryFn: () => api.notifications(query), enabled, staleTime: 30_000, placeholderData: (previous) => previous });

export function useNotificationActions() {
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const settle = () => {
    void client.invalidateQueries({ queryKey: ["notifications", "list"] });
    void client.invalidateQueries({ queryKey: ["notifications", "count"] });
  };
  const onError = (error: Error) => push("error", describe(error));
  return {
    read: useMutation({ mutationFn: api.readNotification, onSuccess: settle, onError }),
    readAll: useMutation({ mutationFn: api.readAllNotifications, onSuccess: settle, onError }),
    remove: useMutation({ mutationFn: api.deleteNotification, onSuccess: settle, onError }),
  };
}
