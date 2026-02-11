"use client";

import { useState } from "react";
import { ChatList } from "@/components/chat/chat-list";
import { ChatWindow } from "@/components/chat/chat-window";
import { PageShell } from "@/components/page-shell";
import { Card } from "@/components/ui/card";
import { MessageSquare } from "lucide-react";
import { useSession, signIn } from "next-auth/react";
import { useSearchParams, useRouter } from "next/navigation";

import { Suspense } from "react";

function ChatPageContent() {
    const { status } = useSession();
    const searchParams = useSearchParams();
    const router = useRouter();

    const conversationId = searchParams.get("id");
    const [selectedConversationId, setSelectedConversationId] = useState<string | null>(conversationId);

    // URL param and state are already synced by initialization and handleSelect
    // No need for redundant useEffect that triggers cascading renders

    const handleSelect = (id: string) => {
        setSelectedConversationId(id);
        router.push(`/chat?id=${id}`);
    };

    if (status === "loading") return null;
    if (status === "unauthenticated") {
        signIn("matcher", { callbackUrl: "/chat" });
        return null;
    }


    return (
        <PageShell
            header={
                <div className="space-y-1">
                    <h1 className="text-2xl font-bold tracking-tight">Messages</h1>
                    <p className="text-muted-foreground">Chat with your matches</p>
                </div>
            }
        >
            <div className="grid grid-cols-1 md:grid-cols-[300px_1fr] gap-6 h-[calc(100vh-200px)]">
                {/* On mobile, hidden if conversation selected? Or list always visible? 
                For MVP, standard 2-col on desktop. Mobile stack is tricky without responsive logic.
                Assuming responsive is handled by columns (stack on mobile). 
                If selected, show window.
            */}
                <Card className={`flex flex-col border-r-0 md:border-r ${selectedConversationId ? 'hidden md:flex' : 'flex'}`}>
                    <div className="p-4 font-semibold border-b">
                        Inbox
                    </div>
                    <ChatList
                        selectedId={selectedConversationId || undefined}
                        onSelect={handleSelect}
                    />
                </Card>

                <Card className={`flex flex-col overflow-hidden ${!selectedConversationId ? 'hidden md:flex' : 'flex'}`}>
                    {selectedConversationId ? (
                        <div className="h-full flex flex-col">
                            {/* Back button for mobile? */}
                            <div className="md:hidden p-2">
                                <button onClick={() => handleSelect("")} className="text-sm text-primary">Back to list</button>
                            </div>
                            <ChatWindow conversationId={selectedConversationId} />
                        </div>
                    ) : (
                        <div className="flex h-full items-center justify-center flex-col gap-4 text-muted-foreground">
                            <MessageSquare className="h-12 w-12 opacity-20" />
                            <p>Select a conversation to start chatting</p>
                        </div>
                    )}
                </Card>
            </div>
        </PageShell>
    );
}

export default function ChatPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <ChatPageContent />
        </Suspense>
    );
}
