"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  RefreshCwIcon,
  ShieldIcon,
} from "lucide-react";
import { Page } from "@/components/page";
import { useQuery, useMutation } from "@apollo/client/react";
import { GET_SPACE, JOIN_SPACE, LEAVE_SPACE } from "@/lib/models/spaces/gql";
import type {
  GetSpaceQuery,
  GetSpaceQueryVariables,
  JoinSpaceMutation,
  JoinSpaceMutationVariables,
  LeaveSpaceMutation,
  LeaveSpaceMutationVariables,
  Member
} from "@/lib/graphql/__generated__/graphql";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MembersDataTable } from "./members/data-table";
import { SpaceHeader } from "@/components/space-header";
import { SpaceSettingsView } from "@/components/spaces/space-settings-view";
import { TierSelectionModal } from "@/components/spaces/tier-selection-modal";
import { CreatePost } from "./feed/create-post";
import { PostList } from "./feed/post-list";
import { PaymentRequiredView } from "@/components/spaces/payment-required-view";

export default function SpaceDetailPage() {
  const params = useParams();
  const router = useRouter();
  const spaceSlug = params.spaceSlug as string;

  const { data, loading, error, refetch } = useQuery<GetSpaceQuery, GetSpaceQueryVariables>(GET_SPACE, {
    variables: { slug: spaceSlug },
    skip: !spaceSlug,
  });

  const [joinSpace, { loading: joining }] = useMutation<JoinSpaceMutation, JoinSpaceMutationVariables>(JOIN_SPACE);
  const [leaveSpace] = useMutation<LeaveSpaceMutation, LeaveSpaceMutationVariables>(LEAVE_SPACE);

  const [feedRefreshTrigger, setFeedRefreshTrigger] = useState(0);
  const [isJoinModalOpen, setIsJoinModalOpen] = useState(false);

  const space = data?.space;

  const handleJoin = async (tierId?: string) => {
    try {
      await joinSpace({
        variables: { spaceSlug, tierId }
      });

      setIsJoinModalOpen(false);
      await refetch();
    } catch (error) {
      console.error("Failed to join space:", error);
      alert("Failed to join space.");
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
      await leaveSpace({
        variables: { spaceId: space!.id }
      });
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

  if (error) {
    console.error("Failed to fetch space:", error);
  }

  if (!space) return null;

  // Show Paywall if waiting for payment
  if (space?.myMembership?.status === "waiting_payment") {
    // Use the tier info from myMembership, or fallback if needed (though query should provide it)
    const tier = space.myMembership.tier;

    return (
      <Page
        breadcrumbs={[
          { label: "Spaces", href: "/spaces" },
          { label: space.name }
        ]}
        header={<SpaceHeader space={space} />}
        actions={
          <Button variant="outline" onClick={handleLeave}>Cancel Request</Button>
        }
      >
        <PaymentRequiredView
          spaceName={space.name}
          tierName={tier?.name}
          price={tier?.price}
          interval={tier?.interval}
          onPaymentComplete={() => refetch()}
        />
      </Page>
    )
  }

  return (
    <>
      <Page
        breadcrumbs={[
          { label: "Spaces", href: "/spaces" },
          { label: space.name }
        ]}
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
                  spaceId={space.id}
                  onPostCreated={() => setFeedRefreshTrigger(prev => prev + 1)}
                />
                <PostList
                  spaceId={space.id}
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
                    spaceId={space.id}
                    onMemberUpdated={() => refetch()}
                    isAdmin={currentUserIsAdmin}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            {currentUserIsAdmin && (
              <TabsContent value="settings" className="mt-6">
                <SpaceSettingsView space={space} onUpdate={() => refetch()} />
              </TabsContent>
            )}
          </Tabs>
        )}
      </Page>

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
