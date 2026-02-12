import { query } from "@/lib/graphql/apollo-client";
import { GET_ALL_SPACES } from "@/lib/models/spaces/gql";
import { GET_DAILY_MATCHES } from "@/lib/models/matches/gql";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { UserCard } from "@/components/user-card";
import { CreateSpaceButton } from "./create-space-button";
import { Sparkles } from "lucide-react";
import type {
  GetAllSpacesQuery,
  GetDailyMatchesQuery
} from "@/lib/graphql/__generated__/graphql";

export default async function DiscoverSpacesPage() {
  const [spacesRes, matchesRes] = await Promise.all([
    query<GetAllSpacesQuery>({ query: GET_ALL_SPACES }),
    query<GetDailyMatchesQuery>({ query: GET_DAILY_MATCHES }).catch(() => ({ data: { dailyMatches: [] } }))
  ]);

  const spaces = spacesRes.data?.spaces ?? [];
  const matches = matchesRes.data?.dailyMatches ?? [];

  return (
    <Page
      breadcrumbs={[
        { label: "Spaces" }
      ]}
      header={
        <div className="space-y-1">
          <h1 className="text-4xl font-extrabold tracking-tight">Discover Spaces</h1>
          <p className="text-lg text-muted-foreground font-medium">Explore and join communities and clubs</p>
        </div>
      }
      actions={<CreateSpaceButton />}
    >
      <div className="space-y-10">
        {/* Top Matches Section */}
        <section className="space-y-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary animate-pulse" />
            <h2 className="text-xl font-bold tracking-tight">Top Matches</h2>
          </div>

          {matches.length === 0 ? (
            <Card className="border-dashed py-8 bg-muted/30">
              <CardContent className="flex flex-col items-center justify-center text-center space-y-2 py-4">
                <p className="text-muted-foreground font-medium">No matches found for today yet.</p>
                <p className="text-xs text-muted-foreground italic">Try updating your profile to improve matching.</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {matches.map((match) => (
                <UserCard
                  key={match.id}
                  user={match}
                  compatibility={0.95 - (matches.indexOf(match) * 0.05)}
                />
              ))}
            </div>
          )}
        </section>

        {/* Discover Spaces Section */}
        <section className="space-y-4">
          <h2 className="text-xl font-bold tracking-tight">Discover Spaces</h2>
          {spaces.length === 0 ? (
            <Card className="text-center py-12 shadow-none border-dashed bg-muted/30">
              <CardHeader>
                <div className="text-6xl mb-4 text-primary">🪐</div>
                <CardTitle className="text-xl">No spaces yet</CardTitle>
                <CardDescription>Create your first space to start building your community</CardDescription>
              </CardHeader>
              <CardContent>
                <CreateSpaceButton />
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {spaces.map((space) => (
                <SpaceCard key={space.id} space={space} />
              ))}
            </div>
          )}
        </section>
      </div>
    </Page>
  );
}
