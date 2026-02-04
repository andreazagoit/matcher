"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, Save, User, UserCircle } from "lucide-react";
import Link from "next/link";

interface UserData {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    birthDate: string;
    gender: "man" | "woman" | "non_binary";
}

export default function AccountPage() {
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [formData, setFormData] = useState<UserData | null>(null);

    useEffect(() => {
        fetchUserData();
    }, []);

    const fetchUserData = async () => {
        try {
            const res = await fetch("/api/auth/account");
            if (!res.ok) {
                if (res.status === 401) {
                    window.location.href = "/api/auth/signin"; // Redirect if not logged in
                    return;
                }
                throw new Error("Failed to load account data");
            }
            const data = await res.json();
            setFormData(data.user);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!formData) return;

        setSaving(true);
        setError(null);
        setSuccess(null);

        try {
            const res = await fetch("/api/auth/account", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || "Failed to update profile");
            }

            setSuccess("Profile updated successfully");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save changes");
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    if (!formData) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Card className="w-full max-w-md">
                    <CardHeader>
                        <CardTitle className="text-destructive">Error</CardTitle>
                        <CardDescription>{error || "Could not load user data"}</CardDescription>
                    </CardHeader>
                    <CardFooter>
                        <Button onClick={() => window.location.reload()}>Retry</Button>
                    </CardFooter>
                </Card>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background p-6">
            <div className="max-w-2xl mx-auto space-y-6">
                {/* Header with Back Link */}
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold flex items-center gap-2">
                        <UserCircle className="w-8 h-8" />
                        Il mio Account
                    </h1>
                    <Link href="/dashboard">
                        <Button variant="outline">Vai alla Dashboard</Button>
                    </Link>
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle>Informazioni Personali</CardTitle>
                        <CardDescription>
                            Gestisci i tuoi dati personali e le preferenze del profilo.
                        </CardDescription>
                    </CardHeader>
                    <form onSubmit={handleSubmit}>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="firstName">Nome</Label>
                                    <Input
                                        id="firstName"
                                        value={formData.firstName}
                                        onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="lastName">Cognome</Label>
                                    <Input
                                        id="lastName"
                                        value={formData.lastName}
                                        onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                                        required
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="email">Email</Label>
                                <Input
                                    id="email"
                                    type="email"
                                    value={formData.email}
                                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    required
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="birthDate">Data di Nascita</Label>
                                    <Input
                                        id="birthDate"
                                        type="date"
                                        value={formData.birthDate ? new Date(formData.birthDate).toISOString().split('T')[0] : ''}
                                        onChange={(e) => setFormData({ ...formData, birthDate: e.target.value })}
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="gender">Genere</Label>
                                    <Select
                                        value={formData.gender}
                                        onValueChange={(val: any) => setFormData({ ...formData, gender: val })}
                                    >
                                        <SelectTrigger>
                                            <SelectValue placeholder="Seleziona genere" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="man">Uomo</SelectItem>
                                            <SelectItem value="woman">Donna</SelectItem>
                                            <SelectItem value="non_binary">Non binario</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>

                            {error && (
                                <div className="p-3 rounded-md bg-destructive/10 text-destructive text-sm font-medium">
                                    {error}
                                </div>
                            )}

                            {success && (
                                <div className="p-3 rounded-md bg-green-500/10 text-green-600 text-sm font-medium">
                                    {success}
                                </div>
                            )}
                        </CardContent>
                        <CardFooter className="flex justify-end">
                            <Button type="submit" disabled={saving}>
                                {saving ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        Salvataggio...
                                    </>
                                ) : (
                                    <>
                                        <Save className="w-4 h-4 mr-2" />
                                        Salva Modifiche
                                    </>
                                )}
                            </Button>
                        </CardFooter>
                    </form>
                </Card>
            </div>
        </div>
    );
}
