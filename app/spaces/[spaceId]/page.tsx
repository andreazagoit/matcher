"use client";

import { useCallback, useEffect, useState, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  CopyIcon,
  RefreshCwIcon,
  TrashIcon,
  SettingsIcon,
} from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { graphql } from "@/lib/graphql/client";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MembersDataTable } from "./members/data-table";
import type { Member } from "./members/columns";
import { CreatePost } from "./feed/create-post";
import { PostList } from "./feed/post-list";

interface Space {
  id: string;
  name: string;
  slug: string;
  description?: string;
  clientId: string;
  // stored locally in session storage only on creation, 
  // but we might want a way to rotate/view it if we keep that logic
  // For now, let's assume we can't view it again unless rotated
  isActive: boolean;
  visibility: string;
  joinPolicy: string;
  membersCount: number;
  createdAt: string;
  members?: Member[];
}

// Member type is defined in ./members/columns.tsx

export default function SpaceDetailPage() {
  const params = useParams();
  const router = useRouter();
  const spaceId = params.spaceId as string;

  const [space, setSpace] = useState<Space | null>(null);
  const [loading, setLoading] = useState(true);
  const { data: session } = useSession();
  const [feedRefreshTrigger, setFeedRefreshTrigger] = useState(0);

  // Check if current user is an admin of this space
  const currentUserIsAdmin = useMemo(() => {
    const userId = session?.user?.id;
    if (!userId || !space?.members) return false;
    const currentMember = space.members.find(m => m.user.id === userId);
    return currentMember?.role === "admin";
  }, [session?.user?.id, space?.members]);

  const fetchSpace = useCallback(async () => {
    try {
      const data = await graphql<{ space: Space }>(`
        query GetSpace($id: ID!) {
          space(id: $id) {
            id
            name
            slug
            description
            clientId
            isActive
            visibility
            joinPolicy
            membersCount
            createdAt
            members(limit: 100) {
              id
              role
              status
              joinedAt
              user {
                id
                firstName
                lastName
                email
              }
            }
          }
        }
      `, { id: spaceId });

      if (data.space) {
        setSpace(data.space);
      } else {
        router.push("/spaces");
      }
    } catch (error) {
      console.error("Failed to fetch space:", error);
    } finally {
      setLoading(false);
    }
  }, [spaceId, router]);

  useEffect(() => {
    fetchSpace();
  }, [fetchSpace]);

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete this space? This cannot be undone.")) return;

    try {
      await graphql(`
        mutation DeleteSpace($id: ID!) {
          deleteSpace(id: $id)
        }
      `, { id: spaceId });

      router.push("/spaces");
    } catch (error) {
      console.error("Failed to delete space", error);
      alert("Failed to delete space");
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
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
      title={space.name}
      subtitle={
        <div className="flex items-center gap-3 mt-1">
          <Badge variant={space.visibility === "public" ? "outline" : "secondary"}>
            {space.visibility === "public" ? "Public" : "Private"}
          </Badge>
          <span className="font-mono bg-muted px-2 py-0.5 rounded text-xs">{space.slug}</span>
          <span className="text-muted-foreground">•</span>
          <span className="text-muted-foreground font-medium">{space.membersCount} members</span>
          {space.description && (
            <>
              <span className="text-muted-foreground">•</span>
              <span className="text-muted-foreground italic font-normal">{space.description}</span>
            </>
          )}
        </div>
      }
      actions={
        <div className="flex gap-2">
          <Link href={`/spaces/${spaceId}/settings`}>
            <Button variant="outline" className="gap-2">
              <SettingsIcon className="h-4 w-4" />
              Settings
            </Button>
          </Link>
          <Button variant="destructive" size="icon" onClick={handleDelete}>
            <TrashIcon className="h-4 w-4" />
          </Button>
        </div>
      }
    >
      <Tabs defaultValue="feed" className="w-full">
        <TabsList className="grid w-full grid-cols-3 max-w-[600px]">
          <TabsTrigger value="feed">Feed</TabsTrigger>
          <TabsTrigger value="members">Members</TabsTrigger>
          <TabsTrigger value="developers">Developers</TabsTrigger>
        </TabsList>

        <TabsContent value="feed" className="mt-6">
          <div className="mx-auto max-w-2xl">
            <CreatePost
              spaceId={spaceId}
              onPostCreated={() => setFeedRefreshTrigger(prev => prev + 1)}
            />
            <PostList
              spaceId={spaceId}
              isAdmin={currentUserIsAdmin}
              refreshTrigger={feedRefreshTrigger}
            />
          </div>
        </TabsContent>

        <TabsContent value="members" className="mt-6">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Members</CardTitle>
                  <CardDescription>Manage community members</CardDescription>
                </div>
                {/* Add Invite Button here later */}
              </div>
            </CardHeader>
            <CardContent>
              <MembersDataTable
                members={(space.members || []) as Member[]}
                spaceId={spaceId}
                onMemberUpdated={fetchSpace}
                isAdmin={currentUserIsAdmin}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="developers" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Developer Credentials</CardTitle>
              <CardDescription>Use these to integrate your custom apps with this Space</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Client ID */}
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
                      onClick={() => copyToClipboard(space.clientId)}
                    >
                      <CopyIcon className="h-4 w-4" />
                    </InputGroupButton>
                  </InputGroupAddon>
                </InputGroup>
              </div>

              <div className="bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 p-4 rounded-lg text-sm">
                Secret keys are only shown once upon creation or rotation. If you lost your secret key, you can generate a new one in Settings.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}
