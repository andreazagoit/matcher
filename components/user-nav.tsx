"use client";

import { useEffect, useState } from "react";
import { signIn } from "next-auth/react";
import Link from "next/link";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { LogIn } from "lucide-react";

interface UserInfo {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
}

export function UserNav() {
    const [user, setUser] = useState<UserInfo | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function checkAuth() {
            try {
                const res = await fetch("/api/auth/profile-status");
                if (res.ok) {
                    const data = await res.json();
                    if (data.authenticated) {
                        setUser(data.user);
                    }
                }
            } catch {
                // Not authenticated
            } finally {
                setLoading(false);
            }
        }
        checkAuth();
    }, []);

    const handleLogin = () => {
        signIn("matcher", { callbackUrl: "/spaces" });
    };

    const handleLogout = async () => {
        await fetch("/api/auth/logout", { method: "POST" });
        window.location.href = "/";
    };

    if (loading) {
        return (
            <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-full bg-muted animate-pulse" />
            </div>
        );
    }

    if (!user) {
        return (
            <Button onClick={handleLogin} size="sm" variant="outline">
                <LogIn className="mr-2 h-4 w-4" />
                Login
            </Button>
        );
    }

    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                    <Avatar className="h-8 w-8">
                        <AvatarFallback>
                            {user.firstName?.[0]}{user.lastName?.[0]}
                        </AvatarFallback>
                    </Avatar>
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[200px]">
                <DropdownMenuLabel className="font-normal">
                    <div className="flex flex-col space-y-1">
                        <p className="text-sm font-medium leading-none">{user.firstName} {user.lastName}</p>
                        <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
                    </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                    <Link href="/account">Profile</Link>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleLogout} className="text-destructive focus:text-destructive">
                    Log out
                </DropdownMenuItem>
            </DropdownMenuContent>
        </DropdownMenu>
    );
}
