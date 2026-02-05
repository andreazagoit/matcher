"use client";

import { useCallback, useEffect, useState } from "react";
import { PageShell } from "@/components/page-shell";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
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
import { AlertCircle, ChevronLeft, RefreshCwIcon, SaveIcon } from "lucide-react";
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

interface Space {
    id: string;
    name: string;
    slug: string;
    description?: string;
    isActive: boolean;
    visibility: string;
    joinPolicy: string;
}

export default function SpaceSettingsPage() {
    const params = useParams();
    const router = useRouter();
    const spaceId = params.spaceId as string;

    const [space, setSpace] = useState<Space | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const [formData, setFormData] = useState({
        name: "",
        slug: "",
        description: "",
        visibility: "public",
        joinPolicy: "open",
    });

    const fetchSpace = useCallback(async () => {
        try {
            const data = await graphql<{ space: Space }>(`
        query GetSpaceSettings($id: ID!) {
          space(id: $id) {
            id
            name
            slug
            description
            isActive
            visibility
            joinPolicy
          }
        }
      `, { id: spaceId });

            if (data.space) {
                setSpace(data.space);
                setFormData({
                    name: data.space.name,
                    slug: data.space.slug,
                    description: data.space.description || "",
                    visibility: data.space.visibility,
                    joinPolicy: data.space.joinPolicy,
                });
            } else {
                router.push("/spaces");
            }
        } catch (error) {
            console.error("Failed to fetch space:", error);
            setError("Failed to load space settings");
        } finally {
            setLoading(false);
        }
    }, [spaceId, router]);

    useEffect(() => {
        fetchSpace();
    }, [fetchSpace]);

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
                id: spaceId,
                input: {
                    name: formData.name,
                    slug: formData.slug || undefined,
                    description: formData.description,
                    visibility: formData.visibility,
                    joinPolicy: formData.joinPolicy,
                }
            });

            setSuccess("Space settings saved successfully");

            // Refresh space data to ensure consistency
            fetchSpace();
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
      `, { id: spaceId });

            router.push("/spaces");
        } catch (error) {
            console.error("Failed to delete space", error);
            setError("Failed to delete space");
            setDeleting(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <RefreshCwIcon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (!space) return null;

    return (
        <PageShell
            breadcrumbs={
                <>
                    <Link href="/spaces" className="hover:text-foreground transition">Spaces</Link>
                    <span>/</span>
                    <Link href={`/spaces/${spaceId}`} className="hover:text-foreground transition">{space.name}</Link>
                    <span>/</span>
                    <span className="text-foreground">Settings</span>
                </>
            }
            title="Space Settings"
            subtitle="Configure your space's visibility, access, and general information"
            actions={
                <Link href={`/spaces/${spaceId}`}>
                    <Button variant="outline" className="gap-2">
                        <ChevronLeft className="w-4 h-4" />
                        Back to Space
                    </Button>
                </Link>
            }
        >
            <div className="max-w-4xl mx-auto space-y-8">
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
        </PageShell>
    );
}
