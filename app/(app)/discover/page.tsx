import { cookies } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_ALL_SPACES, GET_RECOMMENDED_SPACES } from "@/lib/models/spaces/gql";
import { GET_FIND_MATCHES } from "@/lib/models/matches/gql";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { UserCard } from "@/components/user-card";
import { LocationSelector } from "@/components/location-selector";
import { ItemCarousel } from "@/components/item-carousel";
import type {
    GetAllSpacesQuery,
    GetRecommendedSpacesQuery,
    GetRecommendedSpacesQueryVariables,
} from "@/lib/graphql/__generated__/graphql";

const DEFAULT_RADIUS = 50;

interface MatchUser {
    id: string;
    name: string;
    givenName: string;
    familyName: string;
    image: string | null;
    gender: string | null;
    birthdate: string;
}

interface Match {
    user: MatchUser;
    score: number;
    distanceKm: number | null;
    sharedTags: string[];
}

interface FindMatchesQuery {
    findMatches: Match[];
}

export default async function DiscoverPage() {
    const cookieStore = await cookies();
    const radius = Number(cookieStore.get("matcher_radius")?.value) || DEFAULT_RADIUS;

    const [spacesRes, recommendedRes, matchesRes] = await Promise.all([
        query<GetAllSpacesQuery>({ query: GET_ALL_SPACES }),
        query<GetRecommendedSpacesQuery, GetRecommendedSpacesQueryVariables>({
            query: GET_RECOMMENDED_SPACES,
            variables: { limit: 6 },
        }).catch(() => ({ data: { recommendedSpaces: [] } })),
        query<FindMatchesQuery>({
            query: GET_FIND_MATCHES,
            variables: { maxDistance: radius },
        }).catch(() => ({ data: { findMatches: [] } })),
    ]);

    const allSpaces = spacesRes.data?.spaces ?? [];
    const recommended = recommendedRes.data?.recommendedSpaces ?? [];
    const matches = matchesRes.data?.findMatches ?? [];

    const recommendedIds = new Set(recommended.map((s) => s.id));
    const otherSpaces = allSpaces.filter((s) => !recommendedIds.has(s.id));

    return (
        <Page
            breadcrumbs={[{ label: "Discover" }]}
            header={
                <div className="space-y-1">
                    <h1 className="text-6xl font-extrabold tracking-tight">Discover</h1>
                    <p className="text-lg text-muted-foreground font-medium">Explore and join communities and clubs</p>
                </div>
            }
            headerExtras={<LocationSelector />}
        >
            <div className="space-y-12">

                {matches.length === 0 ? (
                    <Card className="border-dashed py-8 bg-muted/30">
                        <CardContent className="flex flex-col items-center justify-center text-center space-y-2 py-4">
                            <p className="text-muted-foreground font-medium">No matches found in {radius} km.</p>
                            <p className="text-xs text-muted-foreground italic">
                                Set your location or increase the radius in the header.
                            </p>
                        </CardContent>
                    </Card>
                ) : (
                    <ItemCarousel title="Daily Matches">
                        {matches.map((match) => (
                            <UserCard
                                key={match.user.id}
                                user={match.user}
                                compatibility={match.score}
                            />
                        ))}
                    </ItemCarousel>
                )}

                {recommended.length > 0 && (
                    <ItemCarousel title="Recommended for you">
                        {recommended.map((space) => (
                            <SpaceCard key={space.id} space={space} />
                        ))}
                    </ItemCarousel>
                )}

                {otherSpaces.length > 0 && (
                    <ItemCarousel title="All Spaces" titleHref="/spaces">
                        {otherSpaces.map((space) => (
                            <SpaceCard key={space.id} space={space} />
                        ))}
                    </ItemCarousel>
                )}

                {allSpaces.length === 0 && (
                    <Card className="text-center py-12 shadow-none border-dashed bg-muted/30">
                        <CardHeader>
                            <div className="text-6xl mb-4 text-primary">🪐</div>
                            <CardTitle className="text-xl">No spaces yet</CardTitle>
                            <CardDescription>No communities available yet. Check back soon!</CardDescription>
                        </CardHeader>
                    </Card>
                )}
            </div>
        </Page>
    );
}
