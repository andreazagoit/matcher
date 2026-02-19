import { query } from "@/lib/graphql/apollo-client";
import { GET_MY_SPACES } from "@/lib/models/spaces/gql";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { CreateSpaceButton } from "./create-space-button";
import { ItemCarousel } from "@/components/item-carousel";
import type { GetMySpacesQuery } from "@/lib/graphql/__generated__/graphql";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default async function MySpacesPage() {
  const spacesRes = await query<GetMySpacesQuery>({ query: GET_MY_SPACES });
  const spaces = spacesRes.data?.mySpaces ?? [];

  return (
    <Page
      breadcrumbs={[{ label: "My Spaces" }]}
      header={
        <div className="space-y-1">
          <h1 className="text-6xl font-extrabold tracking-tight">My Spaces</h1>
          <p className="text-lg text-muted-foreground font-medium">Communities you belong to</p>
        </div>
      }
      actions={<CreateSpaceButton />}
    >
      {spaces.length === 0 ? (
        <Card className="text-center py-12 shadow-none border-dashed bg-muted/30">
          <CardHeader>
            <div className="text-6xl mb-4 text-primary">🪐</div>
            <h3 className="text-xl font-semibold">You haven&apos;t joined any space yet</h3>
            <CardDescription>Explore communities to find where you belong</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <Button asChild size="lg" className="rounded-full px-8">
              <Link href="/discover">Go to Discover</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <ItemCarousel>
          {spaces.map((space) => (
            <SpaceCard key={space.id} space={space} />
          ))}
        </ItemCarousel>
      )}
    </Page>
  );
}
