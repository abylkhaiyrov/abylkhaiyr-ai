import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Download, MoreHorizontal, Pencil, Pin, PinOff, Plus, Search, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { api, type ChatSummary } from "@/lib/api";
import { buildRows, type ChatRow } from "@/lib/chat-groups";
import { cn } from "@/lib/utils";

const HEADER_HEIGHT = 30;
const ROW_HEIGHT = 36;

type SidebarProps = {
  chats: ChatSummary[];
  selected: string | null;
  query: string;
  onQuery: (value: string) => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onPin: (chat: ChatSummary) => void;
  onRename: (chat: ChatSummary, title: string) => void;
  onDelete: (chat: ChatSummary) => void;
};

function RowMenu({ chat, onPin, onRename, onDelete }: { chat: ChatSummary; onPin: () => void; onRename: () => void; onDelete: () => void }) {
  const { t } = useTranslation();
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!anchor) return;
    const close = (event: Event) => {
      if (event.target instanceof Node && (menu.current?.contains(event.target) || trigger.current?.contains(event.target))) return;
      setAnchor(null);
    };
    const key = (event: KeyboardEvent) => event.key === "Escape" && setAnchor(null);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", key);
    window.addEventListener("resize", close);
    document.addEventListener("scroll", close, true);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", key);
      window.removeEventListener("resize", close);
      document.removeEventListener("scroll", close, true);
    };
  }, [anchor]);
  const item = "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm hover:bg-muted";
  const MENU_HEIGHT = 190;
  const top = anchor ? (anchor.bottom + MENU_HEIGHT > window.innerHeight ? anchor.top - MENU_HEIGHT : anchor.bottom + 4) : 0;
  return (
    <>
      <button ref={trigger} type="button" aria-label={t("chat.actions")} aria-haspopup="menu" aria-expanded={anchor !== null} onClick={(event) => { event.stopPropagation(); setAnchor(anchor ? null : event.currentTarget.getBoundingClientRect()); }} className="grid h-7 w-7 place-items-center rounded-md text-muted-foreground hover:bg-background hover:text-foreground">
        <MoreHorizontal size={15} />
      </button>
      {anchor
        ? createPortal(
            <div ref={menu} role="menu" style={{ position: "fixed", top, left: Math.max(8, anchor.right - 176) }} className="z-[70] w-44 rounded-lg border bg-card p-1 shadow-xl animate-rise" onClick={() => setAnchor(null)}>
              <button role="menuitem" type="button" className={item} onClick={onPin}>{chat.pinned ? <PinOff size={14} /> : <Pin size={14} />}{chat.pinned ? t("chat.unpin") : t("chat.pin")}</button>
              <button role="menuitem" type="button" className={item} onClick={onRename}><Pencil size={14} />{t("chat.rename")}</button>
              <a role="menuitem" className={item} href={api.exportChatUrl(chat.id, "md")} download><Download size={14} />{t("chat.exportMd")}</a>
              <a role="menuitem" className={item} href={api.exportChatUrl(chat.id, "json")} download><Download size={14} />{t("chat.exportJson")}</a>
              <button role="menuitem" type="button" className={cn(item, "text-destructive")} onClick={onDelete}><Trash2 size={14} />{t("common.delete")}</button>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}

export function ChatSidebar({ chats, selected, query, onQuery, onSelect, onNew, onPin, onRename, onDelete }: SidebarProps) {
  const { t } = useTranslation();
  const [renaming, setRenaming] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [removing, setRemoving] = useState<ChatSummary | null>(null);
  const parent = useRef<HTMLDivElement>(null);
  const rows = useMemo<ChatRow[]>(() => buildRows(chats), [chats]);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parent.current,
    estimateSize: (index) => (rows[index].type === "header" ? HEADER_HEIGHT : ROW_HEIGHT),
    overscan: 8,
  });
  const commit = (chat: ChatSummary) => {
    if (title.trim() && title.trim() !== chat.title) onRename(chat, title.trim());
    setRenaming(null);
  };
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid gap-2 p-3">
        <Button onClick={onNew} className="w-full justify-start"><Plus size={15} />{t("chat.newChat")}</Button>
        <div className="relative">
          <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input type="search" aria-label={t("chat.search")} placeholder={t("chat.search")} className="field pl-8" value={query} onChange={(event) => onQuery(event.target.value)} />
        </div>
      </div>
      <div ref={parent} className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {rows.length === 0 ? <p className="p-3 text-center text-sm text-muted-foreground">{query ? t("chat.noResults") : t("chat.noChats")}</p> : null}
        <div style={{ height: virtualizer.getTotalSize() }} className="relative w-full">
          {virtualizer.getVirtualItems().map((virtual) => {
            const row = rows[virtual.index];
            const style = { position: "absolute" as const, top: 0, left: 0, width: "100%", height: virtual.size, transform: `translateY(${virtual.start}px)` };
            if (row.type === "header") return <p key={`h-${row.group}`} style={style} className="label-caps flex items-end px-2 pb-1">{t(`chat.groups.${row.group}`)}</p>;
            const chat = row.chat;
            return (
              <div key={chat.id} style={style} className="py-0.5">
                <div className={cn("group flex h-full items-center rounded-lg px-2 transition-colors", chat.id === selected ? "bg-primary/10" : "hover:bg-muted")}>
                  {renaming === chat.id ? (
                    <input autoFocus aria-label={t("chat.rename")} className="h-7 min-w-0 flex-1 rounded border bg-background px-1.5 text-sm" value={title} onChange={(event) => setTitle(event.target.value)} onBlur={() => commit(chat)} onKeyDown={(event) => { if (event.key === "Enter") commit(chat); if (event.key === "Escape") setRenaming(null); }} />
                  ) : (
                    <button type="button" className="flex min-w-0 flex-1 items-center gap-1.5 text-left" onClick={() => onSelect(chat.id)} title={chat.title}>
                      {chat.pinned ? <Pin size={12} className="shrink-0 text-muted-foreground" /> : null}
                      <span className={cn("truncate text-sm", chat.id === selected && "font-medium")}>{chat.title}</span>
                    </button>
                  )}
                  <div className="shrink-0 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
                    <RowMenu chat={chat} onPin={() => onPin(chat)} onRename={() => { setTitle(chat.title); setRenaming(chat.id); }} onDelete={() => setRemoving(chat)} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <ConfirmDialog open={removing !== null} danger title={t("chat.deleteTitle")} description={t("chat.deleteText", { title: removing?.title ?? "" })} confirmLabel={t("common.delete")} onClose={() => setRemoving(null)} onConfirm={() => { if (removing) onDelete(removing); setRemoving(null); }} />
    </div>
  );
}
