"use client";

import { useQuery, useMutation } from "@apollo/client/react";
import { GET_MESSAGES, SEND_MESSAGE } from "@/lib/models/conversations/gql";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Send, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";


interface Participant {
    id: string;
    firstName: string;
    lastName: string;
    image?: string;
}

interface Message {
    id: string;
    content: string;
    createdAt: string;
    sender: Participant;
}

interface Conversation {
    id: string;
    otherParticipant: Participant;
}

interface ChatWindowProps {
    conversationId: string;
}

export function ChatWindow({ conversationId }: ChatWindowProps) {
    const { data: session } = useSession();
    const [content, setContent] = useState("");
    const scrollRef = useRef<HTMLDivElement>(null);

    const { data, loading, error } = useQuery<{ messages: Message[], conversation: Conversation }>(GET_MESSAGES, {
        variables: { conversationId },
        pollInterval: 3000,
    });

    const [sendMessage, { loading: sending }] = useMutation(SEND_MESSAGE, {
        refetchQueries: [GET_MESSAGES, "GetConversations"], // Refetch list too to update sorting
    });

    // Scroll to bottom when messages change
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [data?.messages]);

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!content.trim()) return;

        try {
            await sendMessage({
                variables: { conversationId, content: content.trim() }
            });
            setContent("");
        } catch (err) {
            console.error("Failed to send", err);
        }
    };

    if (loading) return <div className="flex h-full items-center justify-center"><Loader2 className="animate-spin" /></div>;
    if (error) return <div className="flex h-full items-center justify-center text-destructive">Error loading messages</div>;

    const conversation = data?.conversation;
    // Messages are returned newest first (DESC), so we reverse for display if we stack bottom-up
    // But wait, list display usually wants oldest at top, newest at bottom.
    // API returns [Newest, ..., Oldest]
    const messages = [...(data?.messages || [])].reverse();

    return (
        <div className="flex flex-col h-full">
            <div className="flex items-center p-4">
                <div className="flex items-center gap-2">
                    <Avatar className="h-8 w-8">
                        <AvatarImage src={conversation?.otherParticipant?.image} />
                        <AvatarFallback>{conversation?.otherParticipant?.firstName[0]}</AvatarFallback>
                    </Avatar>
                    <div className="font-semibold">
                        {conversation?.otherParticipant?.firstName} {conversation?.otherParticipant?.lastName}
                    </div>
                </div>
            </div>
            <Separator />

            <ScrollArea className="flex-1 p-4">
                <div className="flex flex-col gap-4">
                    {messages.map((msg) => {
                        const isMe = msg.sender.id === session?.user?.id;
                        return (
                            <div
                                key={msg.id}
                                className={`flex ${isMe ? "justify-end" : "justify-start"}`}
                            >
                                <div
                                    className={`max-w-[70%] rounded-lg px-4 py-2 ${isMe
                                        ? "bg-primary text-primary-foreground"
                                        : "bg-muted"
                                        }`}
                                >
                                    {msg.content}
                                </div>
                            </div>
                        );
                    })}
                    <div ref={scrollRef} />
                </div>
            </ScrollArea>

            <div className="p-4 border-t">
                <form onSubmit={handleSend} className="flex gap-2">
                    <Input
                        placeholder="Type a message..."
                        value={content}
                        onChange={(e) => setContent(e.target.value)}
                        disabled={sending}
                    />
                    <Button type="submit" size="icon" disabled={sending || !content.trim()}>
                        <Send className="h-4 w-4" />
                    </Button>
                </form>
            </div>
        </div>
    );
}
