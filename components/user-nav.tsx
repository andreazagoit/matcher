"use client";

import { signOut, useSession } from "@/lib/auth-client";
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

export function UserNav() {
    const { data: session, isPending } = useSession();

    const handleLogout = async () => {
        await signOut({ fetchOptions: { onSuccess: () => { window.location.href = "/"; } } });
    };

    if (isPending) {
        return (
            <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-full bg-muted animate-pulse" />
            </div>
        );
    }

    if (!session?.user) {
        return (
            <Button asChild size="sm" variant="outline">
                <Link href="/sign-in">
                    <LogIn className="mr-2 h-4 w-4" />
                    Login
                </Link>
            </Button>
        );
    }

    const user = session.user;
    const givenName = (user as Record<string, unknown>).givenName as string || "";
    const familyName = (user as Record<string, unknown>).familyName as string || "";

    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                    <Avatar className="h-8 w-8">
                        <AvatarFallback>
                            {givenName?.[0]}{familyName?.[0]}
                        </AvatarFallback>
                    </Avatar>
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[200px]">
                <DropdownMenuLabel className="font-normal">
                    <div className="flex flex-col space-y-1">
                        <p className="text-sm font-medium leading-none">{givenName} {familyName}</p>
                        <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
                    </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                {(() => { const u = (user as Record<string, unknown>).username as string | undefined; return u ? (
                    <DropdownMenuItem asChild>
                        <Link href={`/users/${u}`}>Profile</Link>
                    </DropdownMenuItem>
                ) : null; })()}
                <DropdownMenuItem onClick={handleLogout} className="text-destructive focus:text-destructive">
                    Log out
                </DropdownMenuItem>
            </DropdownMenuContent>
        </DropdownMenu>
    );
}
