"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, RefreshCwIcon, SaveIcon, CopyIcon } from "lucide-react";
import { graphql } from "@/lib/graphql/client";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
    InputGroup,
    InputGroupAddon,
    InputGroupButton,
    InputGroupInput,
} from "@/components/ui/input-group";
import { MembershipTiersManager } from "./settings/membership-tiers-manager";

interface SpaceSettingsViewProps {
    space: {
        id: string;
        name: string;
        slug: string;
        description?: string;
        visibility: string;
        joinPolicy: string;
        clientId?: string;
    };
    onUpdate?: () => void;
}

export function SpaceSettingsView({ space, onUpdate }: SpaceSettingsViewProps) {
    const router = useRouter();
    const [saving, setSaving] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const [formData, setFormData] = useState({
        name: space.name,
        slug: space.slug,
        description: space.description || "",
        visibility: space.visibility,
        joinPolicy: space.joinPolicy,
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setSaving(true);
        setError(null);
        setSuccess(null);

        try {
            await graphql(`
        mutation UpdateSpace($id: ID!, $input: UpdateSpaceInput!) {
          updateSpace(id: $id, input: $input) {
            id
            name
            slug
            description
            visibility
            joinPolicy
          }
        }
      `, {
                id: space.id,
                input: {
                    name: formData.name,
                    slug: formData.slug || undefined,
                    description: formData.description,
                    visibility: formData.visibility,
                    joinPolicy: formData.joinPolicy,
                }
            });

            setSuccess("Space settings saved successfully");
            if (onUpdate) onUpdate();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to update space");
        } finally {
            setSaving(false);
        }
    };

    const handleDelete = async () => {
        setDeleting(true);
        try {
            await graphql(`
        mutation DeleteSpace($id: ID!) {
          deleteSpace(id: $id)
        }
      `, { id: space.id });

            router.push("/spaces");
        } catch (error) {
            console.error("Failed to delete space", error);
            setError("Failed to delete space");
            setDeleting(false);
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        // Maybe add toast here?
    };

    return (
        <div className="space-y-8 w-full max-w-4xl">
            {/* General Settings */}
            <Card>
                <CardHeader>
                    <CardTitle>General Settings</CardTitle>
                    <CardDescription>Update your space information and preferences</CardDescription>
                </CardHeader>
                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-6">
                        {error && (
                            <Alert variant="destructive">
                                <AlertCircle className="h-4 w-4" />
                                <AlertTitle>Error</AlertTitle>
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        )}

                        {success && (
                            <Alert className="border-green-500/50 text-green-600 bg-green-500/10">
                                <AlertTitle>Success</AlertTitle>
                                <AlertDescription>{success}</AlertDescription>
                            </Alert>
                        )}

                        <div className="space-y-2">
                            <Label htmlFor="name">Space Name</Label>
                            <Input
                                id="name"
                                value={formData.name}
                                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                required
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="slug">URL Slug</Label>
                            <Input
                                id="slug"
                                value={formData.slug}
                                onChange={(e) => setFormData({ ...formData, slug: e.target.value })}
                                placeholder="my-space-slug"
                            />
                            <p className="text-xs text-muted-foreground">Unique identifier for your space URL</p>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="description">Description</Label>
                            <Textarea
                                id="description"
                                value={formData.description}
                                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                placeholder="Describe your community..."
                                rows={4}
                            />
                        </div>

                        <Separator />

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <Label htmlFor="visibility">Visibility</Label>
                                <Select
                                    value={formData.visibility}
                                    onValueChange={(value) => setFormData({ ...formData, visibility: value })}
                                >
                                    <SelectTrigger id="visibility">
                                        <SelectValue placeholder="Select visibility" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="public">Public</SelectItem>
                                        <SelectItem value="private">Private</SelectItem>
                                        <SelectItem value="hidden">Hidden</SelectItem>
                                    </SelectContent>
                                </Select>
                                <p className="text-xs text-muted-foreground">Public spaces are discoverable by everyone.</p>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="joinPolicy">Join Policy</Label>
                                <Select
                                    value={formData.joinPolicy}
                                    onValueChange={(value) => setFormData({ ...formData, joinPolicy: value })}
                                >
                                    <SelectTrigger id="joinPolicy">
                                        <SelectValue placeholder="Select policy" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="open">Open</SelectItem>
                                        <SelectItem value="apply">Apply (Req. Approval)</SelectItem>
                                        <SelectItem value="invite_only">Invite Only</SelectItem>
                                    </SelectContent>
                                </Select>
                                <p className="text-xs text-muted-foreground">How users join your space.</p>
                            </div>
                        </div>
                    </CardContent>
                    <CardFooter className="flex justify-end border-t px-6 py-4">
                        <Button type="submit" disabled={saving}>
                            {saving && <RefreshCwIcon className="mr-2 h-4 w-4 animate-spin" />}
                            {!saving && <SaveIcon className="mr-2 h-4 w-4" />}
                            Save Changes
                        </Button>
                    </CardFooter>
                </form>
            </Card>

            {/* Membership Settings - Moved here */}
            <Card>
                <CardContent className="pt-6">
                    <MembershipTiersManager spaceId={space.id} />
                </CardContent>
            </Card>

            {/* Developer Settings (Merged) */}
            {space.clientId && (
                <Card>
                    <CardHeader>
                        <CardTitle>Developer Credentials</CardTitle>
                        <CardDescription>Use these to integrate your custom apps with this Space</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="space-y-2">
                            <Label>Client ID</Label>
                            <InputGroup>
                                <InputGroupInput
                                    value={space.clientId}
                                    readOnly
                                    className="font-mono bg-muted"
                                />
                                <InputGroupAddon align="inline-end">
                                    <InputGroupButton
                                        size="icon-xs"
                                        variant="ghost"
                                        onClick={() => copyToClipboard(space.clientId!)}
                                    >
                                        <CopyIcon className="h-4 w-4" />
                                    </InputGroupButton>
                                </InputGroupAddon>
                            </InputGroup>
                        </div>
                        <div className="bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 p-4 rounded-lg text-sm">
                            Secret keys are only shown once upon creation or rotation. If you lost your secret key, you can generate a new one in Settings (Not implemented yet).
                        </div>
                    </CardContent>
                </Card>
            )}


            {/* Danger Zone */}
            <Card className="border-destructive/50">
                <CardHeader>
                    <CardTitle className="text-destructive">Danger Zone</CardTitle>
                    <CardDescription>
                        Irreversible actions for this space
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center justify-between p-4 border border-destructive/20 rounded-lg bg-destructive/5">
                        <div>
                            <h4 className="font-semibold text-destructive">Delete Space</h4>
                            <p className="text-sm text-muted-foreground">
                                Permanently delete this space and all its data. This cannot be undone.
                            </p>
                        </div>
                        <AlertDialog>
                            <AlertDialogTrigger asChild>
                                <Button variant="destructive">Delete Space</Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                                <AlertDialogHeader>
                                    <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                                    <AlertDialogDescription>
                                        This action cannot be undone. This will permanently delete the
                                        <strong> {space.name} </strong> space and remove all associated data including members, profiles, and settings.
                                    </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                                    <AlertDialogAction onClick={handleDelete} className="bg-destructive hover:bg-destructive/90">
                                        {deleting ? "Deleting..." : "Delete Space"}
                                    </AlertDialogAction>
                                </AlertDialogFooter>
                            </AlertDialogContent>
                        </AlertDialog>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
