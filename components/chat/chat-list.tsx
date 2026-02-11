"use client";

import { useQuery } from "@apollo/client/react";
import { GET_CONVERSATIONS } from "@/lib/models/conversations/gql";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface Participant {
    id: string;
    firstName: string;
    lastName: string;
    image?: string;
}

interface Conversation {
    id: string;
    otherParticipant: Participant;
    lastMessage?: {
        content: string;
        createdAt: string;
    };
    updatedAt: string;
    unreadCount: number;
}

interface ChatListProps {
    selectedId?: string;
    onSelect: (id: string) => void;
}

export function ChatList({ selectedId, onSelect }: ChatListProps) {
    const { data, loading, error } = useQuery<{ conversations: Conversation[] }>(GET_CONVERSATIONS, {
        pollInterval: 5000,
    });

    if (loading) return <div className="p-4 flex justify-center"><Loader2 className="animate-spin" /></div>;
    if (error) return <div className="p-4 text-destructive">Error loading chats</div>;

    const conversations = data?.conversations || [];

    if (conversations.length === 0) {
        return <div className="p-4 text-center text-muted-foreground">No conversations yet</div>;
    }

    return (
        <ScrollArea className="h-full">
            <div className="flex flex-col gap-2 p-4 pt-0">
                {conversations.map((conv) => (
                    <button
                        key={conv.id}
                        onClick={() => onSelect(conv.id)}
                        className={cn(
                            "flex flex-col items-start gap-2 rounded-lg border p-3 text-left text-sm transition-all hover:bg-accent",
                            selectedId === conv.id ? "bg-accent" : "bg-transparent"
                        )}
                    >
                        <div className="flex w-full items-center gap-3">
                            <Avatar>
                                <AvatarImage src={conv.otherParticipant.image} />
                                <AvatarFallback>
                                    {conv.otherParticipant.firstName[0]}
                                    {conv.otherParticipant.lastName[0]}
                                </AvatarFallback>
                            </Avatar>
                            <div className="grid gap-1 flex-1">
                                <div className="font-semibold flex justify-between">
                                    <span>{conv.otherParticipant.firstName} {conv.otherParticipant.lastName}</span>
                                    {conv.unreadCount > 0 && (
                                        <span className="flex h-2 w-2 rounded-full bg-primary" />
                                    )}
                                </div>
                                <div className="line-clamp-1 text-xs text-muted-foreground">
                                    {conv.lastMessage?.content || "No messages yet"}
                                </div>
                            </div>
                        </div>
                    </button>
                ))}
            </div>
        </ScrollArea>
    );
}
