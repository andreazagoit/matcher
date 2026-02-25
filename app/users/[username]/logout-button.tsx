"use client";

import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";
import { signOut } from "@/lib/auth-client";

export function LogoutButton() {
    return (
        <Button
            variant="ghost"
            size="sm"
            onClick={() =>
                signOut({ fetchOptions: { onSuccess: () => { window.location.href = "/"; } } })
            }
        >
            <LogOut className="w-4 h-4" />
            <span className="sr-only sm:not-sr-only sm:ml-2">Logout</span>
        </Button>
    );
}
