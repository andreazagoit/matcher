"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { LogOut } from "lucide-react";
import { signOut } from "@/lib/auth-client";
import { useTranslations } from "next-intl";
import type { User } from "@/lib/graphql/__generated__/graphql";

type UserInfo = Pick<User, "username" | "name" | "email" | "birthdate" | "gender">;

function InfoRow({ label, value }: { label: string; value: string | null | undefined }) {
    return (
        <div className="space-y-0.5">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-sm font-medium">{value ?? "—"}</p>
        </div>
    );
}

export function ProfileSettings({ initialUser }: { initialUser: UserInfo }) {
    const tEnums = useTranslations("enums");

    const birthdate = initialUser.birthdate
        ? new Date(initialUser.birthdate).toLocaleDateString("it-IT", { year: "numeric", month: "long", day: "numeric" })
        : null;

    return (
        <section className="space-y-6">
            <h2 className="text-lg font-semibold tracking-tight">Account</h2>

            <Card className="border-none shadow-sm bg-muted/30">
                <CardHeader>
                    <CardTitle className="text-base">Informazioni personali</CardTitle>
                    <CardDescription>I tuoi dati di account.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <InfoRow label="Username" value={initialUser.username ? `@${initialUser.username}` : null} />
                        <InfoRow label="Email" value={initialUser.email} />
                        <InfoRow label="Nome" value={initialUser.name} />
                        <InfoRow label="Data di nascita" value={birthdate} />
                        <InfoRow
                            label="Genere"
                            value={initialUser.gender ? tEnums(`gender.${initialUser.gender}`) : null}
                        />
                    </div>
                </CardContent>
            </Card>

            <Separator />

            <Card className="border-none shadow-sm bg-muted/30">
                <CardHeader>
                    <CardTitle className="text-base">Sessione</CardTitle>
                    <CardDescription>Disconnettiti dal tuo account su questo dispositivo.</CardDescription>
                </CardHeader>
                <CardFooter className="py-4">
                    <Button
                        variant="destructive"
                        onClick={() =>
                            signOut({ fetchOptions: { onSuccess: () => { window.location.href = "/"; } } })
                        }
                    >
                        <LogOut className="w-4 h-4 mr-2" />
                        Disconnetti
                    </Button>
                </CardFooter>
            </Card>
        </section>
    );
}
