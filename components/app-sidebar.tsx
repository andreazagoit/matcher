"use client"

import {
    GalleryVerticalEnd,
    LayoutDashboard,
    LayoutGrid,
    LifeBuoy,
    Rss,
    Send,
} from "lucide-react"

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
    SidebarRail,
} from "@/components/ui/sidebar"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useSession } from "next-auth/react"
import { NavUser } from "@/components/nav-user"
import { NavSecondary } from "@/components/nav-secondary"

// Navigation Data
const data = {
    user: {
        name: "Guest User",
        email: "guest@example.com",
        avatar: "",
    },
    navMain: [
        {
            title: "Platform",
            url: "#",
            items: [
                {
                    title: "Feed",
                    url: "/feed",
                    icon: Rss,
                },
                {
                    title: "Spaces",
                    url: "/spaces",
                    icon: LayoutGrid,
                },
            ],
        },
        {
            title: "Management",
            url: "#",
            items: [
                {
                    title: "Dashboard",
                    url: "/dashboard",
                    icon: LayoutDashboard,
                },
            ],
        },
    ],
    navSecondary: [
        {
            title: "Support",
            url: "#",
            icon: LifeBuoy,
        },
        {
            title: "Feedback",
            url: "#",
            icon: Send,
        },
    ],
}

interface UserInfo {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
    const pathname = usePathname()
    const { data: session } = useSession()

    // Adapt fetched user to NavUser expected format
    const navUser = session?.user ? {
        name: session.user.name || "User",
        email: session.user.email || "",
        avatar: session.user.image || "",
    } : data.user

    return (
        <Sidebar variant="sidebar" {...props}>
            <SidebarHeader>
                <SidebarMenu>
                    <SidebarMenuItem>
                        <SidebarMenuButton size="lg" asChild>
                            <Link href="/">
                                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                                    <GalleryVerticalEnd className="size-4" />
                                </div>
                                <div className="flex flex-col gap-0.5 leading-none">
                                    <span className="font-semibold">Matcher</span>
                                    <span className="">v1.0.0</span>
                                </div>
                            </Link>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                </SidebarMenu>
            </SidebarHeader>
            <SidebarContent>
                {data.navMain.map((item) => (
                    <SidebarGroup key={item.title}>
                        <SidebarGroupLabel>
                            {item.title}
                        </SidebarGroupLabel>
                        <SidebarGroupContent>
                            <SidebarMenu>
                                {item.items.map((subItem) => (
                                    <SidebarMenuItem key={subItem.title}>
                                        <SidebarMenuButton asChild isActive={pathname === subItem.url}>
                                            <Link href={subItem.url}>
                                                {subItem.icon && <subItem.icon />}
                                                <span>{subItem.title}</span>
                                            </Link>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                ))}
                            </SidebarMenu>
                        </SidebarGroupContent>
                    </SidebarGroup>
                ))}

                <NavSecondary items={data.navSecondary} className="mt-auto" />
            </SidebarContent>
            <SidebarFooter className="pt-0">
                <NavUser user={navUser} />
            </SidebarFooter>
            <SidebarRail />
        </Sidebar>
    )
}
