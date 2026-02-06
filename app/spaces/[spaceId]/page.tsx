"use client";

import { useCallback, useEffect, useState, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { cn } from "@/lib/utils";
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
import { SpaceHeader } from "@/components/space-header";
import { SpaceSettingsView } from "@/components/spaces/space-settings-view";
import type { Member } from "./members/columns";
import { CreatePost } from "./feed/create-post";
import { PostList } from "./feed/post-list";

interface Space {
  id: string;
  name: string;
  slug: string;
  description?: string;
  image?: string;
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
    console.log("DEBUG Admin Check:", { userId, sessionUser: session?.user });
    console.log("DEBUG Space Members:", space?.members);

    if (!userId || !space?.members) return false;
    const currentMember = space.members.find(m => m.user.id === userId);
    console.log("DEBUG Current Member Found:", currentMember);

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
            image
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
      header={<SpaceHeader space={space} />}
      actions={null}
    >
      <Tabs defaultValue="feed" className="w-full">
        <TabsList className="w-fit">
          <TabsTrigger value="feed" className="px-6">Feed</TabsTrigger>
          <TabsTrigger value="members" className="px-6">Members</TabsTrigger>
          {currentUserIsAdmin && <TabsTrigger value="settings" className="px-6">Settings</TabsTrigger>}
        </TabsList>

        <TabsContent value="feed" className="mt-6">
          <div className="">
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

        {currentUserIsAdmin && (
          <TabsContent value="settings" className="mt-6">
            <SpaceSettingsView space={space} onUpdate={fetchSpace} />
          </TabsContent>
        )}
      </Tabs>
    </PageShell>
  );
}
