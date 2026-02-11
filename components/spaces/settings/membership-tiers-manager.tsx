"use client"

import { useState } from "react"
import { useQuery, useMutation } from "@apollo/client/react"
import { GET_SPACE_TIERS, CREATE_TIER, ARCHIVE_TIER } from "@/lib/models/tiers/gql"
import type {
    GetSpaceTiersQuery,
    GetSpaceTiersQueryVariables,
    CreateTierMutation,
    CreateTierMutationVariables,
    ArchiveTierMutation,
    ArchiveTierMutationVariables,
    MembershipTier
} from "@/lib/graphql/__generated__/graphql"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Loader2, Plus, Trash2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface MembershipTiersManagerProps {
    spaceId: string
}

export function MembershipTiersManager({ spaceId }: MembershipTiersManagerProps) {
    const { data, loading, error: queryError, refetch } = useQuery<GetSpaceTiersQuery, GetSpaceTiersQueryVariables>(GET_SPACE_TIERS, {
        variables: { spaceId }
    });

    const [createTier, { loading: createLoading }] = useMutation<CreateTierMutation, CreateTierMutationVariables>(CREATE_TIER);
    const [archiveTier, { loading: archiveLoading }] = useMutation<ArchiveTierMutation, ArchiveTierMutationVariables>(ARCHIVE_TIER);

    const [error, setError] = useState<string | null>(null);
    const [isCreating, setIsCreating] = useState(false);
    const [newTier, setNewTier] = useState({
        name: "",
        description: "",
        price: 0,
        interval: "month",
    });

    const tiers = data?.space?.tiers || [];

    const handleCreate = async () => {
        try {
            await createTier({
                variables: {
                    spaceId,
                    input: {
                        name: newTier.name,
                        description: newTier.description,
                        price: Number(newTier.price) * 100, // Convert to cents
                        interval: newTier.interval,
                    },
                }
            });

            setIsCreating(false);
            setNewTier({ name: "", description: "", price: 0, interval: "month" });
            refetch();
        } catch (error) {
            console.error(error);
            setError("Failed to create tier.");
        }
    }

    const handleArchive = async (id: string) => {
        if (!confirm("Are you sure? This will hide the tier from new members.")) return;
        try {
            await archiveTier({
                variables: { id }
            });
            refetch();
        } catch (error) {
            console.error(error);
            setError("Failed to archive tier.");
        }
    }

    if (queryError) {
        return (
            <Alert variant="destructive">
                <AlertCircle className="mr-2 h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>Failed to load tiers</AlertDescription>
            </Alert>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-medium">Membership Tiers</h3>
                    <p className="text-sm text-muted-foreground">
                        Manage subscription levels for your space.
                    </p>
                </div>
                {!isCreating && (
                    <Button onClick={() => setIsCreating(true)}>
                        <Plus className="mr-2 h-4 w-4" /> Add Tier
                    </Button>
                )}
            </div>

            {error && (
                <Alert variant="destructive">
                    <AlertCircle className="mr-2 h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {isCreating && (
                <Card>
                    <CardHeader>
                        <CardTitle>New Tier</CardTitle>
                        <CardDescription>Define the details for this membership level.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid gap-2">
                            <Label>Name</Label>
                            <Input
                                placeholder="e.g. Premium"
                                value={newTier.name}
                                onChange={(e) => setNewTier({ ...newTier, name: e.target.value })}
                            />
                        </div>
                        <div className="grid gap-2">
                            <Label>Description</Label>
                            <Input
                                placeholder="Benefits..."
                                value={newTier.description}
                                onChange={(e) => setNewTier({ ...newTier, description: e.target.value })}
                            />
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="grid gap-2">
                                <Label>Price (EUR)</Label>
                                <Input
                                    type="number"
                                    min="0"
                                    value={newTier.price}
                                    onChange={(e) => setNewTier({ ...newTier, price: Number(e.target.value) })}
                                />
                            </div>
                            <div className="grid gap-2">
                                <Label>Interval</Label>
                                <Select
                                    value={newTier.interval}
                                    onValueChange={(val) => setNewTier({ ...newTier, interval: val })}
                                >
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="month">Monthly</SelectItem>
                                        <SelectItem value="year">Yearly</SelectItem>
                                        <SelectItem value="one_time">One Time</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                        <div className="flex justify-end gap-2">
                            <Button variant="ghost" onClick={() => setIsCreating(false)}>Cancel</Button>
                            <Button onClick={handleCreate} disabled={createLoading || !newTier.name}>
                                {createLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                                Create Tier
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            )}

            <div className="grid gap-4">
                {loading ? (
                    <div className="text-center py-4 text-muted-foreground">Loading tiers...</div>
                ) : tiers.length === 0 && !isCreating ? (
                    <div className="text-center py-8 border rounded-lg text-muted-foreground">
                        No active membership tiers found.
                    </div>
                ) : (
                    tiers.map((tier: MembershipTier) => (
                        <Card key={tier.id}>
                            <CardContent className="flex items-center justify-between p-6">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h4 className="font-semibold">{tier.name}</h4>
                                        <span className="text-xs bg-secondary px-2 py-0.5 rounded-full">
                                            {(tier.price / 100).toLocaleString("en-EU", { style: "currency", currency: "EUR" })}
                                            /{tier.interval === "one_time" ? "lifetime" : tier.interval}
                                        </span>
                                    </div>
                                    {tier.description && (
                                        <p className="text-sm text-muted-foreground mt-1">{tier.description}</p>
                                    )}
                                </div>
                                <Button
                                    variant="destructive"
                                    size="icon"
                                    onClick={() => handleArchive(tier.id)}
                                    disabled={archiveLoading}
                                >
                                    <Trash2 className="h-4 w-4" />
                                </Button>
                            </CardContent>
                        </Card>
                    ))
                )}
            </div>
        </div>
    )
}
