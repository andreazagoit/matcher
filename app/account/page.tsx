"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PageShell } from "@/components/page-shell";
import { Loader2, Save, UserCircle, LogOut } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";
import { signOut } from "next-auth/react";
import { useRouter } from "next/navigation";

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
        <PageShell
            header={
                <div className="space-y-1">
                    <h1 className="text-4xl font-extrabold tracking-tight text-foreground bg-clip-text">Account Settings</h1>
                    <p className="text-lg text-muted-foreground font-medium">Manage your personal information and profile preferences</p>
                </div>
            }
            actions={
                <div className="flex gap-2">
                    <Link href="/spaces">
                        <Button variant="outline">Back to Spaces</Button>
                    </Link>
                    <Button
                        variant="destructive"
                        onClick={async () => {
                            // Clear IdP session (custom user_id cookie)
                            try {
                                await fetch("/api/auth/logout", { method: "POST" });
                            } catch (e) {
                                console.error("IdP logout error:", e);
                            }
                            // Clear NextAuth session and redirect
                            await signOut({ callbackUrl: "/" });
                        }}
                    >
                        <LogOut className="w-4 h-4 mr-2" />
                        Logout
                    </Button>
                </div>
            }
        >
            <div className="w-full mx-auto space-y-8">
                <Card className="border-none shadow-lg">
                    <CardHeader className="pb-4">
                        <CardTitle className="text-2xl flex items-center gap-2.5">
                            <UserCircle className="w-7 h-7 text-primary" />
                            Personal Information
                        </CardTitle>
                        <CardDescription className="text-base">
                            Update your basic details used across the platform.
                        </CardDescription>
                    </CardHeader>
                    <form onSubmit={handleSubmit}>
                        <CardContent className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2.5">
                                    <Label htmlFor="firstName" className="text-sm font-semibold">First Name</Label>
                                    <Input
                                        id="firstName"
                                        value={formData.firstName}
                                        onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                                        required
                                        className="h-11"
                                    />
                                </div>
                                <div className="space-y-2.5">
                                    <Label htmlFor="lastName" className="text-sm font-semibold">Last Name</Label>
                                    <Input
                                        id="lastName"
                                        value={formData.lastName}
                                        onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                                        required
                                        className="h-11"
                                    />
                                </div>
                            </div>

                            <div className="space-y-2.5">
                                <Label htmlFor="email" className="text-sm font-semibold">Email Address</Label>
                                <Input
                                    id="email"
                                    type="email"
                                    value={formData.email}
                                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    required
                                    className="h-11"
                                />
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2.5">
                                    <Label htmlFor="birthDate" className="text-sm font-semibold">Birth Date</Label>
                                    <Input
                                        id="birthDate"
                                        type="date"
                                        value={formData.birthDate ? new Date(formData.birthDate).toISOString().split('T')[0] : ''}
                                        onChange={(e) => setFormData({ ...formData, birthDate: e.target.value })}
                                        required
                                        className="h-11"
                                    />
                                </div>
                                <div className="space-y-2.5">
                                    <Label htmlFor="gender" className="text-sm font-semibold">Gender</Label>
                                    <Select
                                        value={formData.gender}
                                        onValueChange={(val: UserData["gender"]) => setFormData({ ...formData, gender: val })}
                                    >
                                        <SelectTrigger className="h-11">
                                            <SelectValue placeholder="Select gender" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="man">Man</SelectItem>
                                            <SelectItem value="woman">Woman</SelectItem>
                                            <SelectItem value="non_binary">Non-binary</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>

                            {error && (
                                <div className="p-4 rounded-xl bg-destructive/10 text-destructive text-sm font-medium border border-destructive/20">
                                    {error}
                                </div>
                            )}

                            {success && (
                                <div className="p-4 rounded-xl bg-green-500/10 text-green-600 text-sm font-medium border border-green-500/20">
                                    {success}
                                </div>
                            )}
                        </CardContent>
                        <Separator />
                        <CardFooter className="flex justify-end p-6 bg-muted/5">
                            <Button type="submit" disabled={saving} size="lg" className="px-8">
                                {saving ? (
                                    <>
                                        <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                                        Saving Changes...
                                    </>
                                ) : (
                                    <>
                                        <Save className="w-5 h-5 mr-3" />
                                        Save Changes
                                    </>
                                )}
                            </Button>
                        </CardFooter>
                    </form>
                </Card>
            </div>
        </PageShell>
    );
}
