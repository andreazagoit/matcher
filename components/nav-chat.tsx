"use client"

import { MessageSquare } from "lucide-react"
import {
    SidebarGroup,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarMenuAction,
} from "@/components/ui/sidebar"
import Link from "next/link"
import { useQuery } from "@apollo/client/react"
import { GET_RECENT_CONVERSATIONS } from "@/lib/models/conversations/gql"
import { usePathname } from "next/navigation"

interface NavConversation {
    id: string;
    otherParticipant: {
        firstName: string;
        lastName: string;
    };
    unreadCount: number;
}

export function NavChat() {
    const pathname = usePathname()
    const { data } = useQuery<{ conversations: NavConversation[] }>(GET_RECENT_CONVERSATIONS, {
        pollInterval: 10000,
    })

    const conversations = data?.conversations || []

    return (
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
            <SidebarGroupLabel>Messages</SidebarGroupLabel>
            <SidebarMenu>
                {conversations.map((item) => (
                    <SidebarMenuItem key={item.id}>
                        <SidebarMenuButton asChild isActive={pathname === `/chat` && pathname.includes(item.id)}>
                            {/* TODO: Link to specific chat if we had a route /chat/[id], for now just /chat handles selection state internally or via query param? 
                                User asked for sidebar integration. Usually implies navigation.
                                I'll link to /chat?id=... or assumes /chat is the page. 
                                Since /chat page manages selection state, linking to /chat puts us there. 
                                But passing ID would be better. */}
                            <Link href={`/chat?id=${item.id}`}>
                                <span>{item.otherParticipant.firstName} {item.otherParticipant.lastName}</span>
                            </Link>
                        </SidebarMenuButton>
                        {item.unreadCount > 0 && (
                            <SidebarMenuAction className="pointer-events-none">
                                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-[10px] text-primary-foreground">
                                    {item.unreadCount}
                                </span>
                            </SidebarMenuAction>
                        )}
                    </SidebarMenuItem>
                ))}
                <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                        <Link href="/chat">
                            <MessageSquare />
                            <span>All Messages</span>
                        </Link>
                    </SidebarMenuButton>
                </SidebarMenuItem>
            </SidebarMenu>
        </SidebarGroup>
    )
}
