import { notFound } from "next/navigation";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import { query } from "@/lib/graphql/apollo-client";
import { GET_SPACE } from "@/lib/models/spaces/gql";
import type { GetSpaceQuery, GetSpaceQueryVariables, Member } from "@/lib/graphql/__generated__/graphql";
import { Page } from "@/components/page";
import { SpaceHeader } from "@/components/space-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MembersDataTable } from "./members/data-table";
import { SpaceSettingsView } from "@/components/spaces/space-settings-view";
import { FeedSection } from "./feed/feed-section";
import { EventList } from "./events/event-list";
import { PaymentRequiredView } from "@/components/spaces/payment-required-view";
import { ShieldIcon } from "lucide-react";
import { JoinLeaveActions } from "./join-leave-actions";

interface Props {
  params: Promise<{ spaceSlug: string }>;
}

export default async function SpaceDetailPage({ params }: Props) {
  const { spaceSlug } = await params;

  const [sessionRes, spaceRes] = await Promise.all([
    auth.api.getSession({ headers: await headers() }).catch(() => null),
    query<GetSpaceQuery, GetSpaceQueryVariables>({
      query: GET_SPACE,
      variables: { slug: spaceSlug },
    }),
  ]);

  const space = spaceRes.data?.space;
  if (!space) notFound();

  const isAuthenticated = !!sessionRes?.user;
  const isMember = !!space.myMembership;
  const currentUserIsAdmin = space.myMembership?.role === "admin" || space.myMembership?.role === "owner";

  const breadcrumbs = [
    { label: "Spaces", href: "/spaces" },
    { label: space.name },
  ];

  const spaceHeaderProps = { ...space, createdAt: space.createdAt as string | undefined };

  if (space.myMembership?.status === "waiting_payment") {
    const tier = space.myMembership.tier;
    return (
      <Page
        breadcrumbs={breadcrumbs}
        header={<SpaceHeader space={spaceHeaderProps} />}
        actions={
          <JoinLeaveActions
            spaceSlug={spaceSlug}
            spaceId={space.id}
            isMember
            tiers={[]}
            isAuthenticated
            isWaitingPayment
          />
        }
      >
        <PaymentRequiredView
          spaceName={space.name}
          tierName={tier?.name}
          price={tier?.price}
          interval={tier?.interval}
        />
      </Page>
    );
  }

  return (
    <Page
      breadcrumbs={breadcrumbs}
      header={<SpaceHeader space={spaceHeaderProps} />}
      actions={
        <JoinLeaveActions
          spaceSlug={spaceSlug}
          spaceId={space.id}
          isMember={isMember}
          tiers={space.tiers ?? []}
          isAuthenticated={isAuthenticated}
        />
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
            <TabsTrigger value="events" className="px-6">Events</TabsTrigger>
            <TabsTrigger value="members" className="px-6">Members</TabsTrigger>
            {currentUserIsAdmin && <TabsTrigger value="settings" className="px-6">Settings</TabsTrigger>}
          </TabsList>

          <TabsContent value="feed" className="mt-6">
            <FeedSection spaceId={space.id} isAdmin={currentUserIsAdmin} />
          </TabsContent>

          <TabsContent value="events" className="mt-6">
            <EventList spaceId={space.id} isAdmin={currentUserIsAdmin} />
          </TabsContent>

          <TabsContent value="members" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Members</CardTitle>
                <CardDescription>Manage community members</CardDescription>
              </CardHeader>
              <CardContent>
                <MembersDataTable
                  members={(space.members || []) as Member[]}
                  spaceId={space.id}
                  isAdmin={currentUserIsAdmin}
                />
              </CardContent>
            </Card>
          </TabsContent>

          {currentUserIsAdmin && (
            <TabsContent value="settings" className="mt-6">
              <SpaceSettingsView space={space} />
            </TabsContent>
          )}
        </Tabs>
      )}
    </Page>
  );
}
