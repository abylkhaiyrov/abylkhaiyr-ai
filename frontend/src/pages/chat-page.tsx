import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { EmptyState, ErrorState, LoadingState } from "@/components/data-state";

export function ChatPage() {
  const client = useQueryClient();
  const [selectedChat, setSelectedChat] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const chats = useQuery({ queryKey: ["chats"], queryFn: () => api.chats() });
  const chat = useQuery({ queryKey: ["chat", selectedChat], queryFn: () => api.chat(selectedChat ?? ""), enabled: selectedChat !== null });
  const create = useMutation({ mutationFn: api.createChat, onSuccess: (created) => { client.invalidateQueries({ queryKey: ["chats"] }); setSelectedChat(created.id); } });
  const send = useMutation({ mutationFn: () => api.sendMessage(selectedChat ?? "", prompt), onSuccess: () => { setPrompt(""); client.invalidateQueries({ queryKey: ["chat", selectedChat] }); client.invalidateQueries({ queryKey: ["chats"] }); } });
  useEffect(() => { if (!selectedChat && chats.data?.[0]) setSelectedChat(chats.data[0].id); }, [chats.data, selectedChat]);
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); if (prompt.trim() && selectedChat) send.mutate(); }
  if (chats.isPending) return <LoadingState />;
  if (chats.isError) return <ErrorState error={chats.error} retry={() => chats.refetch()} />;
  return <div className="grid min-h-[calc(100vh-8rem)] gap-4 lg:grid-cols-[17rem_1fr]"><aside className="rounded-lg border"><div className="border-b p-3"><Button className="w-full" onClick={() => create.mutate()} disabled={create.isPending}>New chat</Button></div><nav aria-label="Chat history" className="grid gap-1 p-2">{chats.data.map((item) => <button key={item.id} className={`truncate rounded px-3 py-2 text-left text-sm ${selectedChat === item.id ? "bg-muted" : "hover:bg-muted"}`} onClick={() => setSelectedChat(item.id)}>{item.title}</button>)}</nav></aside><section className="flex min-h-0 flex-col rounded-lg border">{selectedChat === null ? <div className="m-auto"><EmptyState title="No conversation selected" description="Create a local chat to begin." /></div> : chat.isPending ? <div className="p-4"><LoadingState /></div> : chat.isError ? <div className="p-4"><ErrorState error={chat.error} retry={() => chat.refetch()} /></div> : <><header className="border-b px-4 py-3"><h1 className="font-semibold">{chat.data.title}</h1><p className="text-xs text-muted-foreground">{chat.data.model_name ?? "No model name recorded"}</p></header><div className="flex-1 space-y-4 overflow-auto p-4">{chat.data.messages.length === 0 ? <EmptyState title="Start the conversation" description="Messages remain local and are not used for training." /> : chat.data.messages.map((message) => <article key={message.id} className={`max-w-3xl rounded-lg p-3 text-sm ${message.role === "user" ? "ml-auto bg-primary text-primary-foreground" : "bg-muted"}`}><p className="mb-1 text-xs opacity-70">{message.role}</p><div className="whitespace-pre-wrap">{message.content}</div>{message.tokens_per_second ? <p className="mt-2 text-xs opacity-70">{message.tokens_per_second.toFixed(1)} tok/s</p> : null}</article>)}</div><form className="flex gap-2 border-t p-3" onSubmit={submit}><label className="sr-only" htmlFor="chat-prompt">Message</label><textarea id="chat-prompt" className="min-h-10 flex-1 resize-none rounded-md border bg-transparent p-2 text-sm" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Send a message" /><Button type="submit" disabled={send.isPending || !prompt.trim()}>{send.isPending ? "Generating…" : "Send"}</Button></form>{send.error ? <p role="alert" className="px-3 pb-3 text-sm text-destructive">{send.error.message}</p> : null}</>}</section></div>;
}
