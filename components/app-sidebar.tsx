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
} from "@/components/ui/sidebar"
import { LayoutDashboard, Compass, Plus, Rss, LayoutGrid } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState, useEffect } from "react"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

interface UserInfo {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
}

export function AppSidebar() {
    const pathname = usePathname()
    const [user, setUser] = useState<UserInfo | null>(null)

    useEffect(() => {
        async function checkAuth() {
            try {
                const res = await fetch("/api/auth/profile-status")
                if (res.ok) {
                    const data = await res.json()
                    if (data.authenticated) {
                        setUser(data.user)
                    }
                }
            } catch {
                // Not authenticated
            }
        }
        checkAuth()
    }, [])

    return (
        <Sidebar>
            <SidebarHeader className="h-16 border-b flex items-center justify-center px-4 shrink-0">
                <Link href="/spaces" className="flex items-center gap-2 font-semibold w-full">
                    <span>Matcher</span>
                </Link>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel>Application</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild isActive={pathname === "/feed"}>
                                    <Link href="/feed">
                                        <Rss />
                                        <span>Feed</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton asChild isActive={pathname === "/spaces"}>
                                    <Link href="/spaces">
                                        <LayoutGrid />
                                        <span>Spaces</span>
                                    </Link>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                <SidebarGroup>
                    <SidebarGroupLabel>Management</SidebarGroupLabel>
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
            <SidebarFooter className="border-t p-4">
                {user ? (
                    <Link
                        href="/account"
                        className="flex items-center gap-3 px-2 py-1 rounded-lg hover:bg-accent transition-colors group"
                    >
                        <Avatar className="h-9 w-9 border-2 border-transparent group-hover:border-primary/20 transition-all">
                            <AvatarFallback className="bg-primary/10 text-primary font-medium">
                                {user.firstName?.[0]}{user.lastName?.[0]}
                            </AvatarFallback>
                        </Avatar>
                        <div className="flex flex-col min-w-0">
                            <span className="text-sm font-semibold truncate text-foreground">
                                {user.firstName} {user.lastName}
                            </span>
                            <span className="text-xs text-muted-foreground truncate">
                                View Profile
                            </span>
                        </div>
                    </Link>
                ) : (
                    <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
                        <Compass className="h-3 w-3" />
                        <span>Matcher v1.0</span>
                    </div>
                )}
            </SidebarFooter>
        </Sidebar>
    )
}
