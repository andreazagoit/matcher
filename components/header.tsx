"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { LogIn } from "lucide-react";
import { useEffect, useState } from "react";
import { signIn } from "next-auth/react";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface UserInfo {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
}

export function Header() {
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
        // Use Auth.js signIn - it handles PKCE automatically
        signIn("matcher", { callbackUrl: "/dashboard" });
    };

    const handleLogout = async () => {
        await fetch("/api/auth/logout", { method: "POST" });
        window.location.href = "/";
    };

    return (
        <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                        <span className="text-primary-foreground font-bold text-sm">M</span>
                    </div>
                    <span className="text-xl font-bold text-foreground">Matcher</span>
                </Link>

                <nav className="flex items-center gap-6">
                    <Link href="/dashboard" className="text-muted-foreground hover:text-foreground transition">
                        Dashboard
                    </Link>
                    <Link href="/docs" className="text-muted-foreground hover:text-foreground transition">
                        Docs
                    </Link>

                    {loading ? (
                        <div className="w-8 h-8 rounded-full bg-muted animate-pulse" />
                    ) : user ? (
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary cursor-pointer hover:bg-primary/30 transition-colors">
                                    <span className="text-sm font-medium">
                                        {user.firstName?.[0]}{user.lastName?.[0]}
                                    </span>
                                </div>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                                <DropdownMenuLabel>{user.firstName} {user.lastName}</DropdownMenuLabel>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem asChild>
                                    <Link href="/account" className="cursor-pointer">Account</Link>
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={handleLogout} className="cursor-pointer">
                                    Logout
                                </DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>
                    ) : (
                        <Button size="sm" onClick={handleLogin}>
                            <LogIn className="w-4 h-4 mr-2" />
                            Login
                        </Button>
                    )}
                </nav>
            </div>
        </header>
    );
}
