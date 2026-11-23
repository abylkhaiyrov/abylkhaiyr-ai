import { useMemo } from "react";
import { useInfiniteQuery, type QueryKey } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState } from "@/components/data-state";
import { Button } from "@/components/ui/button";
import type { Page } from "@/lib/api";
import { nextOffset } from "@/lib/history";

type Props<T> = {
  queryKey: QueryKey;
  load: (offset: number) => Promise<Page<T>>;
  empty: string;
  children: (items: T[], total: number) => React.ReactNode;
  staleTime?: number;
};

export function DataPager<T>({ queryKey, load, empty, children, staleTime = 30_000 }: Props<T>) {
  const { t } = useTranslation();
  const query = useInfiniteQuery({ queryKey, queryFn: ({ pageParam }) => load(pageParam), initialPageParam: 0, getNextPageParam: (_last, pages) => nextOffset(pages), staleTime });
  const items = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data]);
  const total = query.data?.pages.at(-1)?.total ?? 0;
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  if (items.length === 0) return <EmptyState title={empty} />;
  return (
    <>
      {children(items, total)}
      {query.hasNextPage ? <Button variant="outline" className="justify-self-center" disabled={query.isFetchingNextPage} onClick={() => void query.fetchNextPage()}>{t("common.showMore")}</Button> : null}
    </>
  );
}
