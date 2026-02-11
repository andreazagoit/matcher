"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, Save } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { useMutation } from "@apollo/client/react";
import { UPDATE_USER } from "@/lib/models/users/gql";
import type { User, UpdateUserMutation, UpdateUserMutationVariables } from "@/lib/graphql/__generated__/graphql";
import { Gender } from "@/lib/graphql/__generated__/graphql";

// Pick only the fields we need for the form
type UserFormData = Pick<User, "id" | "firstName" | "lastName" | "email" | "birthDate" | "gender">;

export function AccountForm({ initialUser }: { initialUser: UserFormData }) {
    const [updateUser, { loading: mutationLoading }] = useMutation<UpdateUserMutation, UpdateUserMutationVariables>(UPDATE_USER);

    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const [formData, setFormData] = useState({
        ...initialUser,
        gender: initialUser.gender || Gender.Man,
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

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
            // No need to refetch if we update optimistic/local state or if server response updates cache,
            // but since parent is Server Component, we might want to refresh router to update server data?
            // For now, simple success message is enough as form is updated.
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save changes");
        }
    };

    return (
        <Card className="border-none shadow-lg">
            <CardHeader className="pb-4">
                <CardTitle className="text-2xl flex items-center gap-2.5">
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
                                onChange={(e) => setFormData(prev => ({ ...prev, firstName: e.target.value }))}
                                required
                                className="h-11"
                            />
                        </div>
                        <div className="space-y-2.5">
                            <Label htmlFor="lastName" className="text-sm font-semibold">Last Name</Label>
                            <Input
                                id="lastName"
                                value={formData.lastName}
                                onChange={(e) => setFormData(prev => ({ ...prev, lastName: e.target.value }))}
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
                            onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
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
                                // Ensure date format YYYY-MM-DD
                                value={formData.birthDate ? new Date(formData.birthDate).toISOString().split('T')[0] : ''}
                                onChange={(e) => setFormData(prev => ({ ...prev, birthDate: e.target.value }))}
                                required
                                className="h-11"
                            />
                        </div>
                        <div className="space-y-2.5">
                            <Label htmlFor="gender" className="text-sm font-semibold">Gender</Label>
                            <Select
                                value={formData.gender}
                                onValueChange={(val: Gender) => setFormData(prev => ({ ...prev, gender: val }))}
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
    );
}
