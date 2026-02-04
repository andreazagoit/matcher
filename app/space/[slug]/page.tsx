"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { graphql } from "@/lib/graphql/client";
import { Loader2Icon, UserIcon, UsersIcon } from "lucide-react";
import Link from "next/link";
import { useSession } from "next-auth/react";

interface Space {
    id: string;
    name: string;
    slug: string;
    description?: string;
    membersCount: number;
    isPublic: boolean;
    requiresApproval: boolean;
    myMembership?: {
        role: string;
        status: string;
    };
}

export default function PublicSpacePage() {
    const params = useParams();
    const slug = params.slug as string;
    const router = useRouter();
    const { data: session } = useSession();

    const [space, setSpace] = useState<Space | null>(null);
    const [loading, setLoading] = useState(true);
    const [joining, setJoining] = useState(false);

    const fetchSpace = async () => {
        try {
            const data = await graphql<{ space: Space }>(`
        query GetPublicSpace($slug: String!) {
          space(slug: $slug) {
            id
            name
            slug
            description
            membersCount
            isPublic
            requiresApproval
            myMembership {
              role
              status
            }
          }
        }
      `, { slug });

            if (data.space) {
                setSpace(data.space);
            } else {
                // Handle 404
            }
        } catch (error) {
            console.error("Failed to fetch space:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchSpace();
    }, [slug]);

    const handleJoin = async () => {
        if (!session) {
            // Redirect to login
            router.push(`/api/auth/signin?callbackUrl=/space/${slug}`);
            return;
        }

        if (!space) return;

        setJoining(true);
        try {
            await graphql(`
        mutation JoinSpace($spaceId: ID!) {
          joinSpace(spaceId: $spaceId) {
            id
            status
          }
        }
      `, { spaceId: space.id });

            // Refresh space data to update UI
            fetchSpace();
        } catch (error) {
            console.error("Failed to join space:", error);
            alert("Failed to join space");
        } finally {
            setJoining(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (!space) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen gap-4">
                <h1 className="text-4xl font-bold">404</h1>
                <p className="text-muted-foreground">Space not found</p>
                <Link href="/">
                    <Button variant="outline">Go Home</Button>
                </Link>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background flex flex-col items-center py-20 px-4">
            <Card className="w-full max-w-2xl shadow-lg border-2">
                <CardHeader className="text-center pb-8 pt-10">
                    <div className="w-24 h-24 bg-primary/10 rounded-3xl mx-auto flex items-center justify-center mb-6 text-4xl text-primary font-bold">
                        {space.name.charAt(0).toUpperCase()}
                    </div>
                    <CardTitle className="text-3xl mb-2">{space.name}</CardTitle>
                    {space.description && (
                        <CardDescription className="text-lg max-w-md mx-auto">
                            {space.description}
                        </CardDescription>
                    )}

                    <div className="flex justify-center gap-4 mt-6">
                        <Badge variant="secondary" className="px-3 py-1">
                            <UsersIcon className="h-3 w-3 mr-1" />
                            {space.membersCount} members
                        </Badge>
                        <Badge variant={space.isPublic ? "outline" : "secondary"}>
                            {space.isPublic ? "Public Group" : "Private Group"}
                        </Badge>
                    </div>
                </CardHeader>

                <CardContent className="flex flex-col items-center pb-10">
                    {!space.myMembership ? (
                        <Button
                            size="lg"
                            className="w-full max-w-sm text-lg h-12"
                            onClick={handleJoin}
                            disabled={joining}
                        >
                            {joining ? (
                                <>
                                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                                    Joining...
                                </>
                            ) : (
                                "Join Space"
                            )}
                        </Button>
                    ) : (
                        <div className="text-center space-y-4">
                            <div className="bg-green-50 text-green-700 px-6 py-3 rounded-full font-medium inline-flex items-center">
                                You are a {space.myMembership.role}
                            </div>
                            {/* Future: Feed would go here */}
                            <div className="p-8 border border-dashed rounded-lg bg-muted/50 w-full">
                                <p className="text-muted-foreground">Feed coming soon...</p>
                            </div>
                        </div>
                    )}

                    {!session && (
                        <p className="text-xs text-muted-foreground mt-4">
                            You need to be logged in to join.
                        </p>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
