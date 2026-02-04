"use client"

import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarTrigger,
} from "@/components/ui/sidebar"
import { LayoutDashboard, Settings, Compass, Plus, Rss } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

interface UserInfo {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
}

export function AppSidebar() {
    const pathname = usePathname()

    return (
        <Sidebar>
            <SidebarHeader className="h-16 border-b flex items-center justify-center px-4 shrink-0">
                <Link href="/dashboard" className="flex items-center gap-2 font-semibold w-full">
                    <span>Matcher</span>
                </Link>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel>Application</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild isActive={pathname === "/dashboard"}>
                                    <Link href="/dashboard">
                                        <LayoutDashboard />
                                        <span>Dashboard</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild isActive={pathname === "/dashboard/feed"}>
                                    <Link href="/dashboard/feed">
                                        <Rss />
                                        <span>Feed</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>

                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                <SidebarGroup>
                    <SidebarGroupLabel>Actions</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {/* Placeholder for dynamic space list or actions */}
                            <SidebarMenuItem>
                                <SidebarMenuButton disabled>
                                    <Plus />
                                    <span>Create Space</span>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

            </SidebarContent>
            <SidebarFooter className="p-4">
                {/* Footer content if needed, e.g. Matcher Logo or version */}
                <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
                    <Compass className="h-3 w-3" />
                    <span>Matcher v1.0</span>
                </div>
            </SidebarFooter>
        </Sidebar>
    )
}
