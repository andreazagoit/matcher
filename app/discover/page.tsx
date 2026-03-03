import { cookies } from "next/headers";
import { query } from "@/lib/graphql/apollo-client";
import { GET_DAILY_MATCHES } from "@/lib/models/matches/gql";
import { GET_RECOMMENDED_CATEGORIES, GET_RECOMMENDED_SPACES, GET_USER_RECOMMENDED_EVENTS } from "@/lib/models/users/gql";
import { Card, CardContent } from "@/components/ui/card";
import { Page } from "@/components/page";
import { SpaceCard } from "@/components/spaces/space-card";
import { UserCard } from "@/components/user-card";
import { LocationSelector } from "@/components/location-selector";
import { LocationBanner } from "@/components/location-banner";
import { ItemCarousel } from "@/components/item-carousel";
import { EventCard } from "@/components/event-card";

// ── Types ─────────────────────────────────────────────────────────────────────

interface RecommendedEvent {
    id: string;
    title: string;
    description?: string | null;
    location?: string | null;
    startsAt: string;
    endsAt?: string | null;
    attendeeCount: number;
    categories: string[];
    spaceId: string;
}

interface CategoryWithEvents {
    id: string;
    recommendedEvents: RecommendedEvent[];
}

interface RecommendedSpace {
    id: string;
    name: string;
    slug: string;
    description?: string | null;
    image?: string | null;
    categories: string[];
    visibility: string;
    joinPolicy: string;
    createdAt: string;
    isActive?: boolean | null;
    membersCount?: number | null;
    type?: string | null;
    stripeAccountEnabled: boolean;
}

interface DailyMatch {
    score: number;
    distanceKm?: number;
    user: {
        id: string;
        username: string;
        name: string;
        image?: string;
        gender?: string;
        birthdate: string;
        userItems: { id: string; type: string; content: string; displayOrder: number }[];
    };
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default async function DiscoverPage() {
    const cookieStore = await cookies();
    const hasLocation = !!cookieStore.get("matcher_lat")?.value;

    const [matchesRes, spacesRes, eventsRes, categoriesRes] = await Promise.all([
        query<{ me: { dailyMatches: DailyMatch[] } | null }>({
            query: GET_DAILY_MATCHES,
        }).catch(() => ({ data: { me: null } })),

        query<{ me: { recommendedSpaces: RecommendedSpace[] } | null }>({
            query: GET_RECOMMENDED_SPACES,
            variables: { limit: 8 },
        }).catch(() => ({ data: { me: null } })),

        query<{ me: { recommendedEvents: RecommendedEvent[] } | null }>({
            query: GET_USER_RECOMMENDED_EVENTS,
            variables: { limit: 8 },
        }).catch(() => ({ data: { me: null } })),

        query<{ me: { recommendedCategories: CategoryWithEvents[] } | null }>({
            query: GET_RECOMMENDED_CATEGORIES,
            variables: { limit: 12 },
        }).catch(() => ({ data: { me: null } })),
    ]);

    const matches = matchesRes.data?.me?.dailyMatches ?? [];
    const recommendedSpaces = spacesRes.data?.me?.recommendedSpaces ?? [];
    const recommendedEvents = eventsRes.data?.me?.recommendedEvents ?? [];
    const topCategories: CategoryWithEvents[] = (categoriesRes.data?.me?.recommendedCategories ?? [])
        .filter((cat) => cat.recommendedEvents.length > 0);

    return (
        <Page
            breadcrumbs={[{ label: "Discover" }]}
            header={
                <div className="space-y-1">
                    <h1 className="text-6xl font-extrabold tracking-tight">Discover</h1>
                    <p className="text-lg text-muted-foreground font-medium">
                        Persone, eventi e spazi consigliati per te
                    </p>
                </div>
            }
            headerExtras={<LocationSelector />}
        >
            <div className="space-y-12">

                {!hasLocation && <LocationBanner />}

                {/* ── Daily Matches ─────────────────────────────────────── */}
                {matches.length === 0 ? (
                    <Card className="border-dashed py-8 bg-muted/30">
                        <CardContent className="flex flex-col items-center justify-center text-center space-y-2 py-4">
                            <p className="text-muted-foreground font-medium">Nessun match disponibile oggi.</p>
                            <p className="text-xs text-muted-foreground italic">
                                Imposta la tua posizione o torna più tardi.
                            </p>
                        </CardContent>
                    </Card>
                ) : (
                    <ItemCarousel title="Daily Matches" columns={4}>
                        {matches.map((match) => (
                            <UserCard
                                key={match.user.id}
                                user={match.user}
                                compatibility={match.score}
                            />
                        ))}
                    </ItemCarousel>
                )}

                {/* ── Recommended Spaces ────────────────────────────────── */}
                {recommendedSpaces.length > 0 && (
                    <ItemCarousel title="Spazi consigliati" titleHref="/spaces" columns={4}>
                        {recommendedSpaces.map((space) => (
                            <SpaceCard key={space.id} space={space} />
                        ))}
                    </ItemCarousel>
                )}

                {/* ── Recommended Events ────────────────────────────────── */}
                {recommendedEvents.length > 0 && (
                    <ItemCarousel title="Eventi consigliati" titleHref="/events" columns={4}>
                        {recommendedEvents.map((event) => (
                            <EventCard
                                key={event.id}
                                event={{ ...event, tags: event.categories }}
                            />
                        ))}
                    </ItemCarousel>
                )}

                {/* ── Top 4 Categories → Events ─────────────────────────── */}
                {topCategories.map((cat) => (
                    <ItemCarousel
                        key={cat.id}
                        title={cat.id.charAt(0).toUpperCase() + cat.id.slice(1)}
                        titleHref={`/categories/${cat.id}`}
                        columns={4}
                    >
                        {cat.recommendedEvents.map((event) => (
                            <EventCard
                                key={event.id}
                                event={{ ...event, tags: event.categories }}
                            />
                        ))}
                    </ItemCarousel>
                ))}

            </div>
        </Page>
    );
}
