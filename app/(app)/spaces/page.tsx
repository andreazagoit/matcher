import { query } from "@/lib/graphql/apollo-client";
import { GET_MY_SPACES } from "@/lib/models/spaces/gql";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { CreateSpaceButton } from "./create-space-button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { GetMySpacesQuery } from "@/lib/graphql/__generated__/graphql";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default async function MySpacesPage() {
  const spacesRes = await query<GetMySpacesQuery>({ query: GET_MY_SPACES });
  const allSpaces = spacesRes.data?.mySpaces ?? [];

  const ownedSpaces = allSpaces.filter(s => s.myMembership?.role === 'admin');
  const participatingSpaces = allSpaces; // All spaces I'm in

  return (
    <Page
      breadcrumbs={[
        { label: "My Spaces" }
      ]}
      header={
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">My Spaces</h1>
          <p className="text-lg text-muted-foreground font-medium">Communities you belong to or manage</p>
        </div>
      }
      actions={<CreateSpaceButton />}
    >
      <Tabs defaultValue="participating" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 max-w-[400px]">
          <TabsTrigger value="participating">Participating ({participatingSpaces.length})</TabsTrigger>
          <TabsTrigger value="owned">Owned ({ownedSpaces.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="participating" className="space-y-6">
          {participatingSpaces.length === 0 ? (
            <EmptySpacesState title="You haven't joined any space yet" />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {participatingSpaces.map((space) => (
                <SpaceCard key={space.id} space={space} />
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="owned" className="space-y-6">
          {ownedSpaces.length === 0 ? (
            <EmptySpacesState
              title="You don't own any space yet"
              description="Create your own community and start inviting people"
            />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {ownedSpaces.map((space) => (
                <SpaceCard key={space.id} space={space} />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </Page>
  );
}

function EmptySpacesState({ title, description }: { title: string; description?: string }) {
  return (
    <Card className="text-center py-12 shadow-none border-dashed bg-muted/30">
      <CardHeader>
        <div className="text-6xl mb-4 text-primary">🪐</div>
        <CardTitle className="text-xl">{title}</CardTitle>
        <CardDescription>
          {description || "Explore our communities to find where you belong"}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-4">
        <Button asChild size="lg" className="rounded-full px-8">
          <Link href="/discover">Go to Discover</Link>
        </Button>
        <span className="text-sm text-muted-foreground">or</span>
        <CreateSpaceButton />
      </CardContent>
    </Card>
  );
}
