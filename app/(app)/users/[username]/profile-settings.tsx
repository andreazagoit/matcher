"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { LogOut } from "lucide-react";
import { signOut } from "@/lib/auth-client";
import type { User } from "@/lib/graphql/__generated__/graphql";

type UserInfo = Pick<User, "username" | "givenName" | "familyName" | "email" | "birthdate" | "gender">;

const GENDER_LABEL: Record<string, string> = {
    man: "Man",
    woman: "Woman",
    non_binary: "Non-binary",
};

function InfoRow({ label, value }: { label: string; value: string | null | undefined }) {
    return (
        <div className="space-y-0.5">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-sm font-medium">{value ?? "—"}</p>
        </div>
    );
}

export function ProfileSettings({ initialUser }: { initialUser: UserInfo }) {
    const birthdate = initialUser.birthdate
        ? new Date(initialUser.birthdate).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
        : null;

    return (
        <section className="space-y-6">
            <h2 className="text-lg font-semibold tracking-tight">Account</h2>

            <Card className="border-none shadow-sm bg-muted/30">
                <CardHeader>
                    <CardTitle className="text-base">Personal Information</CardTitle>
                    <CardDescription>Your account details as registered.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <InfoRow label="Username" value={initialUser.username ? `@${initialUser.username}` : null} />
                        <InfoRow label="Email" value={initialUser.email} />
                        <InfoRow label="Given Name" value={initialUser.givenName} />
                        <InfoRow label="Family Name" value={initialUser.familyName} />
                        <InfoRow label="Birth Date" value={birthdate} />
                        <InfoRow label="Gender" value={initialUser.gender ? GENDER_LABEL[initialUser.gender] ?? initialUser.gender : null} />
                    </div>
                </CardContent>
            </Card>

            <Separator />

            <Card className="border-none shadow-sm bg-muted/30">
                <CardHeader>
                    <CardTitle className="text-base">Session</CardTitle>
                    <CardDescription>Log out from your account on this device.</CardDescription>
                </CardHeader>
                <CardFooter className="py-4">
                    <Button
                        variant="destructive"
                        onClick={() =>
                            signOut({ fetchOptions: { onSuccess: () => { window.location.href = "/"; } } })
                        }
                    >
                        <LogOut className="w-4 h-4 mr-2" />
                        Log out
                    </Button>
                </CardFooter>
            </Card>
        </section>
    );
}
