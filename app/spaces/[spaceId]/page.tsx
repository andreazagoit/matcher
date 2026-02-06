"use client";

import { useCallback, useEffect, useState } from "react";

import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  RefreshCwIcon,
  ShieldIcon,
} from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { graphql } from "@/lib/graphql/client";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MembersDataTable } from "./members/data-table";
import { SpaceHeader } from "@/components/space-header";
import { SpaceSettingsView } from "@/components/spaces/space-settings-view";
import { TierSelectionModal } from "@/components/spaces/tier-selection-modal";
import { CreatePost } from "./feed/create-post";
import { PostList } from "./feed/post-list";
import type { Member } from "./members/columns";
import { PaymentRequiredView } from "@/components/spaces/payment-required-view";

interface Tier {
  id: string;
  name: string;
  description?: string;
  price: number;
  interval: string;
  isActive: boolean;
}

interface Space {
  id: string;
  name: string;
  slug: string;
  description?: string;
  image?: string;
  clientId: string;
  isActive: boolean;
  visibility: string;
  joinPolicy: string;
  membersCount: number;
  createdAt: string;
  members?: Member[];
  tiers?: Tier[];
  myMembership?: {
    id: string;
    role: string;
    status: string;
    tier?: Tier;
  } | null;
}

export default function SpaceDetailPage() {
  const params = useParams();
  const router = useRouter();
  const spaceId = params.spaceId as string;

  const [space, setSpace] = useState<Space | null>(null);
  const [loading, setLoading] = useState(true);
  // removed duplicate
  const [feedRefreshTrigger, setFeedRefreshTrigger] = useState(0);
  const [isJoinModalOpen, setIsJoinModalOpen] = useState(false);
  const [joining, setJoining] = useState(false);

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
            myMembership {
                id
                role
                status
                tier {
                    id
                    name
                    price
                    interval
                }
            }
            tiers {
                id
                name
                description
                price
                interval
                isActive
            }
            members(limit: 20) {
              id
              role
              status
              joinedAt
              tier {
                  name
              }
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

  const handleJoin = async (tierId?: string) => {
    setJoining(true);
    try {
      await graphql(`
            mutation JoinSpace($spaceId: ID!, $tierId: ID) {
                joinSpace(spaceId: $spaceId, tierId: $tierId) {
                    id
                    status
                }
            }
          `, { spaceId, tierId });

      setIsJoinModalOpen(false);
      await fetchSpace();
    } catch (error) {
      console.error("Failed to join space:", error);
      alert("Failed to join space.");
    } finally {
      setJoining(false);
    }
  };

  const onJoinClick = () => {
    if (space?.tiers && space.tiers.length > 0) {
      setIsJoinModalOpen(true);
    } else {
      handleJoin();
    }
  }

  const handleLeave = async () => {
    if (!confirm("Are you sure you want to leave this space?")) return;
    try {
      await graphql(`mutation LeaveSpace($spaceId: ID!) { leaveSpace(spaceId: $spaceId) }`, { spaceId });
      router.push("/spaces");
    } catch (err) {
      console.error(err);
      alert("Failed to leave space");
    }
  }

  const currentUserIsAdmin = space?.myMembership?.role === "admin" || space?.myMembership?.role === "owner";
  const isMember = !!space?.myMembership;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <RefreshCwIcon className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!space) return null;

  // Show Paywall if waiting for payment
  if (space?.myMembership?.status === "waiting_payment") {
    // Use the tier info from myMembership, or fallback if needed (though query should provide it)
    const tier = space.myMembership.tier;

    return (
      <PageShell header={<SpaceHeader space={space} />} actions={
        <Button variant="outline" onClick={handleLeave}>Cancel Request</Button>
      }>
        <PaymentRequiredView
          spaceName={space.name}
          tierName={tier?.name}
          price={tier?.price}
          interval={tier?.interval}
          onPaymentComplete={() => fetchSpace()}
        />
      </PageShell>
    )
  }

  return (
    <>
      <PageShell
        header={<SpaceHeader space={space} />}
        actions={
          isMember ? (
            <Button variant="outline" onClick={handleLeave}>Leave Space</Button>
          ) : (
            <Button onClick={onJoinClick} disabled={joining}>
              {joining ? "Joining..." : "Join Space"}
            </Button>
          )
        }
      >
        {!isMember ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="rounded-full bg-muted/20 p-4 mb-4">
              <ShieldIcon className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold">Join to View Content</h3>
            <p className="text-muted-foreground max-w-sm mt-2">
              You need to join this space to view its feed, members, and other content.
            </p>
          </div>
        ) : (
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
        )}
      </PageShell>

      <TierSelectionModal
        isOpen={isJoinModalOpen}
        onClose={() => setIsJoinModalOpen(false)}
        tiers={space.tiers?.filter(t => t.isActive) || []}
        onSelect={handleJoin}
        isJoining={joining}
      />
    </>
  );
}
