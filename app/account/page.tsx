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
import { useQuery, useMutation } from "@apollo/client/react";
import gql from "graphql-tag";

const GET_ME = gql`
  query GetMe {
    me {
      id
      firstName
      lastName
      email
      birthDate
      gender
    }
  }
`;

const UPDATE_USER = gql`
  mutation UpdateUser($id: ID!, $input: UpdateUserInput!) {
    updateUser(id: $id, input: $input) {
      id
      firstName
      lastName
      email
      birthDate
      gender
    }
  }
`;

interface UserData {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    birthDate: string;
    gender: "man" | "woman" | "non_binary";
}

export default function AccountPage() {
    const { data, loading: queryLoading, error: queryError, refetch } = useQuery<{ me: UserData }>(GET_ME);
    const [updateUser, { loading: mutationLoading }] = useMutation<{ updateUser: UserData }, { id: string; input: any }>(UPDATE_USER);

    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [formData, setFormData] = useState<UserData | null>(null);

    useEffect(() => {
        if (!queryLoading && !data?.me && !queryError) {
            window.location.href = "/api/auth/signin";
            return;
        }

        if (data?.me) {
            setFormData({
                id: data.me.id,
                firstName: data.me.firstName,
                lastName: data.me.lastName,
                email: data.me.email,
                birthDate: data.me.birthDate,
                gender: data.me.gender
            });
        }
    }, [data, queryLoading, queryError]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!formData) return;

        setError(null);
        setSuccess(null);

        try {
            const { id, ...input } = formData;
            await updateUser({
                variables: {
                    id,
                    input: {
                        firstName: input.firstName,
                        lastName: input.lastName,
                        email: input.email,
                        birthDate: input.birthDate,
                        gender: input.gender,
                    }
                }
            });

            setSuccess("Profile updated successfully");
            refetch();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save changes");
        }
    };

    if (queryLoading && !formData) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    if (queryError || (!queryLoading && !formData)) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Card className="w-full max-w-md">
                    <CardHeader>
                        <CardTitle className="text-destructive">Error</CardTitle>
                        <CardDescription>{queryError?.message || "Could not load user data"}</CardDescription>
                    </CardHeader>
                    <CardFooter>
                        <Button onClick={() => refetch()}>Retry</Button>
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
                                        value={formData?.firstName || ""}
                                        onChange={(e) => setFormData(prev => prev ? { ...prev, firstName: e.target.value } : null)}
                                        required
                                        className="h-11"
                                    />
                                </div>
                                <div className="space-y-2.5">
                                    <Label htmlFor="lastName" className="text-sm font-semibold">Last Name</Label>
                                    <Input
                                        id="lastName"
                                        value={formData?.lastName || ""}
                                        onChange={(e) => setFormData(prev => prev ? { ...prev, lastName: e.target.value } : null)}
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
                                    value={formData?.email || ""}
                                    onChange={(e) => setFormData(prev => prev ? { ...prev, email: e.target.value } : null)}
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
                                        value={formData?.birthDate ? new Date(formData.birthDate).toISOString().split('T')[0] : ''}
                                        onChange={(e) => setFormData(prev => prev ? { ...prev, birthDate: e.target.value } : null)}
                                        required
                                        className="h-11"
                                    />
                                </div>
                                <div className="space-y-2.5">
                                    <Label htmlFor="gender" className="text-sm font-semibold">Gender</Label>
                                    <Select
                                        value={formData?.gender || ""}
                                        onValueChange={(val: UserData["gender"]) => setFormData(prev => prev ? { ...prev, gender: val } : null)}
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
                            <Button type="submit" disabled={mutationLoading} size="lg" className="px-8">
                                {mutationLoading ? (
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
