"use client";

import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";
import Link from "next/link";
import { signOut } from "@/lib/auth-client";

export function AccountHeaderActions() {
    return (
        <div className="flex gap-2">
            <Link href="/spaces">
                <Button variant="outline">Back to Spaces</Button>
            </Link>
            <Button
                variant="destructive"
                onClick={() => signOut({ fetchOptions: { onSuccess: () => { window.location.href = "/"; } } })}
            >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
            </Button>
        </div>
    );
}
