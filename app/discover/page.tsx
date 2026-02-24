import { cookies } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_ALL_SPACES, GET_RECOMMENDED_SPACES } from "@/lib/models/spaces/gql";
import { GET_FIND_MATCHES } from "@/lib/models/matches/gql";
import { GET_RECOMMENDED_TAGS } from "@/lib/models/users/gql";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { UserCard } from "@/components/user-card";
import { LocationSelector } from "@/components/location-selector";
import { ItemCarousel } from "@/components/item-carousel";
import { getTranslations } from "next-intl/server";
import { Tag } from "lucide-react";
import type {
    GetAllSpacesQuery,
    GetFindMatchesQuery,
    GetFindMatchesQueryVariables,
    GetRecommendedSpacesQuery,
    GetRecommendedSpacesQueryVariables,
    GetRecommendedTagsQuery,
    GetRecommendedTagsQueryVariables,
} from "@/lib/graphql/__generated__/graphql";

const DEFAULT_RADIUS = 50;

export default async function DiscoverPage() {
    const cookieStore = await cookies();
    const radius = Number(cookieStore.get("matcher_radius")?.value) || DEFAULT_RADIUS;

    const [spacesRes, recommendedRes, matchesRes, tagsRes, tTags] = await Promise.all([
        query<GetAllSpacesQuery>({ query: GET_ALL_SPACES }),
        query<GetRecommendedSpacesQuery, GetRecommendedSpacesQueryVariables>({
            query: GET_RECOMMENDED_SPACES,
            variables: { limit: 6 },
        }).catch(() => ({ data: { recommendedSpaces: [] } })),
        query<GetFindMatchesQuery, GetFindMatchesQueryVariables>({
            query: GET_FIND_MATCHES,
            variables: { maxDistance: radius },
        }).catch(() => ({ data: { findMatches: [] as GetFindMatchesQuery["findMatches"] } })),
        query<GetRecommendedTagsQuery, GetRecommendedTagsQueryVariables>({
            query: GET_RECOMMENDED_TAGS,
            variables: { limit: 12 },
        }).catch(() => ({ data: { me: null } })),
        getTranslations("tags"),
    ]);

    const allSpaces = spacesRes.data?.spaces ?? [];
    const recommended = recommendedRes.data?.recommendedSpaces ?? [];
    const matches = matchesRes.data?.findMatches ?? [];
    const recommendedTags = tagsRes.data?.me?.recommendedUserTags ?? [];

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

                {recommendedTags.length > 0 && (
                    <section className="space-y-3">
                        <div className="flex items-center gap-2">
                            <Tag className="h-4 w-4 text-muted-foreground" />
                            <h2 className="text-lg font-semibold tracking-tight">Tag consigliati per te</h2>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {recommendedTags.map((tag) => (
                                <span
                                    key={tag}
                                    className="inline-flex items-center rounded-full border bg-card px-4 py-1.5 text-sm font-medium text-foreground shadow-sm"
                                >
                                    {tTags(tag as Parameters<typeof tTags>[0])}
                                </span>
                            ))}
                        </div>
                    </section>
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
